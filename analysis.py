"""GitHub commit ingestion and chronological regression evaluation."""
from pathlib import Path
from io import BytesIO
import csv
import io
import json
import zipfile
import urllib.request
import urllib.parse
import warnings
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, PoissonRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.exceptions import ConvergenceWarning

ALIASES = {'repo':['repo','repository','repo_name'], 'date':['committer_datetime_utc','committed_at','commit_date','date','author_datetime_utc','timestamp'], 'ai':['ai_assisted','is_ai_assisted'], 'author':['author_id','author_hash','author'], 'type':['author_type'], 'sha':['commit_hash','commit_sha','sha']}
FEATURES=['지난달 AI 표시 비율','지난달 개발자 수(log)','지난달 커밋 수(log)','시간 흐름']

def request(url, token=''):
    headers={'User-Agent':'yulstar-regression','Accept':'application/vnd.github+json'}
    # Only send GitHub credentials to its API, never arbitrary raw file hosts.
    if token and urllib.parse.urlparse(url).hostname=='api.github.com':headers['Authorization']='Bearer '+token
    try:
        with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=120) as r:return r.read()
    except Exception as e:raise ValueError('깃허브 파일을 읽지 못했습니다. 공개 여부·주소·API 사용 한도를 확인하세요. 같은 저장소에 데이터를 넣으면 API 없이 읽을 수 있습니다.') from e

def github_files(repo_url, branch='', token=''):
    p=urllib.parse.urlparse(repo_url.strip())
    if p.hostname!='github.com':raise ValueError('https://github.com/사용자/저장소 형식으로 입력하세요.')
    parts=p.path.strip('/').split('/')
    if len(parts)!=2:raise ValueError('파일 주소가 아닌 저장소 첫 화면 주소를 입력하세요. 브랜치는 별도 칸에 입력하세요.')
    repo='/'.join(parts);api='https://api.github.com/repos/'+repo
    if not branch:branch=json.loads(request(api,token))['default_branch']
    tree=json.loads(request(api+'/git/trees/'+urllib.parse.quote(branch,safe='')+'?recursive=1',token))
    if tree.get('truncated'):raise ValueError('파일 목록이 너무 커 일부가 생략됐습니다. 필요한 데이터를 앱과 같은 저장소에 넣어 주세요.')
    files=[]
    for item in tree.get('tree',[]):
        name=item['path']
        if item['type']=='blob' and Path(name).suffix.lower() in ['.csv','.zip']:
            files.append((name, f'https://raw.githubusercontent.com/{repo}/{urllib.parse.quote(branch,safe="")}/{urllib.parse.quote(name,safe="/")}'))
    return files

def local_files(root):
    return sorted(p for p in Path(root).rglob('*') if p.is_file() and p.suffix.lower() in ['.csv','.zip'] and not any(x.startswith('.') or x in ['node_modules','venv','__pycache__'] for x in p.relative_to(root).parts))

def sources(name, payload):
    if payload[:100].startswith(b'version https://git-lfs.github.com/spec'):
        raise ValueError(f'{name}: 실제 데이터 대신 Git LFS 포인터가 있습니다. 실제 CSV/ZIP이 배포 환경에 내려받아져야 합니다.')
    if name.lower().endswith('.zip'):
        with zipfile.ZipFile(BytesIO(payload)) as z:
            members=[i for i in z.infolist() if not i.is_dir() and i.filename.lower().endswith('.csv') and not i.filename.startswith('__MACOSX/')]
            if sum(i.file_size for i in members)>2_000_000_000:raise ValueError('압축을 푼 CSV가 2GB를 넘습니다. 필요한 파일만 선택해 주세요.')
            for item in members:
                with z.open(item) as f:yield name+' / '+item.filename,f
    else:yield name,BytesIO(payload)

def aggregate(items, progress=None):
    groups={};seen=set();counts=dict(total=0,invalid=0,duplicates=0,bots=0,unknown_ai=0,missing_author=0,no_hash=0)
    columns={};used=[];skipped=[]
    for name,payload in items:
        for label,stream in sources(name,payload):
            if progress:progress(label)
            text=io.TextIOWrapper(stream,encoding='utf-8-sig',newline='')
            reader=csv.DictReader(text);headers=reader.fieldnames or []
            mapping={k:next((a for a in opts if a in headers),None) for k,opts in ALIASES.items()}
            if not all(mapping[k] for k in ['repo','date','ai','author']):
                skipped.append(label);continue
            used.append(label)
            for row in reader:
                if not any(row.values()):continue
                if None in row or any(row.get(h) is None for h in headers):raise ValueError(f'{label}: 열 수가 맞지 않는 CSV 행이 있습니다. 파일 형식을 확인하세요.')
                counts['total']+=1
                for h in headers:
                    c=columns.setdefault(h,{'present':0,'blank':0,'numeric':True});c['present']+=1;v=(row[h] or '').strip()
                    if not v:c['blank']+=1
                    elif c['numeric']:
                        try:float(v)
                        except ValueError:c['numeric']=False
                repo=row[mapping['repo']].strip();raw=row[mapping['date']].strip()
                # ISO UTC source timestamps; reject missing/invalid dates explicitly.
                try:
                    from datetime import datetime, timezone
                    d=datetime.fromisoformat(raw.replace('Z','+00:00'))
                    if d.tzinfo:d=d.astimezone(timezone.utc)
                    month=d.strftime('%Y-%m')
                except (ValueError,TypeError):counts['invalid']+=1;continue
                if not repo:counts['invalid']+=1;continue
                sha=row.get(mapping['sha'],'')
                if sha:
                    key=(repo,sha)
                    if key in seen:counts['duplicates']+=1;continue
                    seen.add(key)
                else:counts['no_hash']+=1
                tp=row.get(mapping['type'],'').strip().lower()
                if 'bot' in tp and 'ai' not in tp:counts['bots']+=1;continue
                g=groups.setdefault((repo,month),{'repo':repo,'month':month,'commits':0,'ai':0,'known':0,'authors':set()});g['commits']+=1
                flag=row[mapping['ai']].strip().lower()
                if flag in ['true','1','yes','false','0','no']:
                    g['known']+=1;g['ai']+=int(flag in ['true','1','yes'])
                else:counts['unknown_ai']+=1
                author=row[mapping['author']].strip()
                if not author:counts['missing_author']+=1
                elif not tp or tp=='human':g['authors'].add(author)
    if not used:raise ValueError('커밋 CSV를 찾지 못했습니다. repo, committer_datetime_utc, ai_assisted, author_id 열이 있는 파일을 선택하세요. repos.csv와 releases.csv만으로는 분석할 수 없습니다.')
    rows=[]
    for g in groups.values():g['developers']=len(g.pop('authors'));rows.append(g)
    if not rows:raise ValueError('처리 후 사용할 커밋이 없습니다.')
    overview=pd.DataFrame([{'열 이름':h,'자료형':'숫자' if c['numeric'] else '문자 / 날짜','빈 값 수':c['blank']+counts['total']-c['present']} for h,c in columns.items()])
    counts['retained']=counts['total']-counts['invalid']-counts['duplicates']-counts['bots']
    return pd.DataFrame(rows).sort_values(['month','repo']).reset_index(drop=True),counts,overview,used,skipped

def examples(monthly):
    current=monthly.copy();prior=monthly.copy()
    prior['month']=(pd.PeriodIndex(prior['month'],freq='M')+1).astype(str)
    prior=prior.rename(columns={c:'prev_'+c for c in ['commits','ai','known','developers']})
    df=current.merge(prior,on=['repo','month'],how='inner',validate='one_to_one')
    df=df[(df.prev_commits>0)&(df.prev_known>0)].copy()
    df[FEATURES[0]]=df.prev_ai/df.prev_known
    df[FEATURES[1]]=np.log1p(df.prev_developers)
    df[FEATURES[2]]=np.log1p(df.prev_commits)
    periods=pd.PeriodIndex(df.month,freq='M');df[FEATURES[3]]=periods.year*12+periods.month
    return df

# Candidate ranges are explicit and finite: lowest validation error within this set.
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit

SEARCH_SPACES = {
    '선형 회귀': [{'linearregression__fit_intercept':[True,False]}],
    '포아송 회귀': [{'poissonregressor__alpha':[0.0,0.001,0.01,0.1,1.0]}],
    '랜덤 포레스트': [
        {'n_estimators':[100,200], 'max_depth':[6,12,None],
         'min_samples_leaf':[2,5], 'max_features':[0.8,1.0]},
        {'n_estimators':[150], 'max_depth':[12],
         'min_samples_leaf':[5], 'max_features':[0.8]},
    ],
}

def chronological_folds(train):
    months=np.array(sorted(train.month.unique()))
    if len(months)<8:
        raise ValueError('자동 비교에는 학습 기간이 최소 8개월 필요합니다. 분석 기간을 늘려 주세요.')
    folds=[];description=[]
    for a,b in TimeSeriesSplit(n_splits=3).split(months):
        ti=np.flatnonzero(train.month.isin(months[a]).to_numpy())
        vi=np.flatnonzero(train.month.isin(months[b]).to_numpy())
        if len(ti)<5 or len(vi)<2:raise ValueError('검증 구간의 데이터가 부족합니다. 분석 기간을 늘려 주세요.')
        folds.append((ti,vi))
        description.append({'회차':len(folds),'학습 시작':months[a[0]],'학습 끝':months[a[-1]],
                            '검증 시작':months[b[0]],'검증 끝':months[b[-1]],'학습 행':len(ti),'검증 행':len(vi)})
    return folds,description

def fit_checked(model,X,y):
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        model.fit(X,y)
    return model

def evaluate(monthly,start,end,ratio=.8,objective='MSE',progress=None):
    if objective not in ('MAE','MSE'):raise ValueError('선택 기준은 MSE 또는 MAE여야 합니다.')
    df=examples(monthly);df=df[df.month.between(start,end)].sort_values(['month','repo']).reset_index(drop=True)
    months=sorted(df.month.unique())
    if len(months)<10:raise ValueError('자동 설정 비교에는 연결된 기록이 있는 달이 최소 10개 필요합니다. 기간을 늘려 주세요.')
    cut=months[min(len(months)-1,max(1,int(len(months)*ratio)))];train=df[df.month<cut].reset_index(drop=True);test=df[df.month>=cut].reset_index(drop=True)
    if len(train)<20 or len(test)<5:raise ValueError('학습용 20행, 평가용 5행 이상이 필요합니다.')
    folds,fold_info=chronological_folds(train)
    models={'선형 회귀':make_pipeline(StandardScaler(),LinearRegression()),
            '포아송 회귀':make_pipeline(StandardScaler(),PoissonRegressor(max_iter=3000)),
            '랜덤 포레스트':RandomForestRegressor(random_state=42,n_jobs=2)}
    X=train[FEATURES];y=train.commits
    trials=[];chosen=[];errors=[];fitted={}
    total=sum(len(list(ParameterGrid(SEARCH_SPACES[n]))) for n in models);done=0
    for name,template in models.items():
        candidates=[]
        for candidate_id,params in enumerate(ParameterGrid(SEARCH_SPACES[name]),1):
            record={'모델':name,'후보':candidate_id,'설정':json.dumps(params,ensure_ascii=False,sort_keys=True),
                    '선택됨':False,'상태':'성공'}
            maes=[];mses=[]
            try:
                for fold_id,(ti,vi) in enumerate(folds,1):
                    model=fit_checked(clone(template).set_params(**params),X.iloc[ti],y.iloc[ti])
                    p=model.predict(X.iloc[vi])
                    if not np.isfinite(p).all():raise ValueError('유효하지 않은 예측값')
                    mae=float(mean_absolute_error(y.iloc[vi],p));mse=float(mean_squared_error(y.iloc[vi],p))
                    maes.append(mae);mses.append(mse)
                    record[f'검증{fold_id} MAE']=mae;record[f'검증{fold_id} MSE']=mse
                record['검증 평균 MAE']=float(np.mean(maes));record['검증 평균 MSE']=float(np.mean(mses))
                candidates.append((record['검증 평균 '+objective],candidate_id,params,record))
            except Exception as exc:
                record['상태']='제외: '+type(exc).__name__
                record['검증 평균 MAE']=np.nan;record['검증 평균 MSE']=np.nan
            trials.append(record);done+=1
            if progress:progress(done,total,f'{name} · {candidate_id}번 설정 비교 완료')
        if not candidates:
            errors.append(f'{name}: 모든 후보가 학습에 실패하여 제외했습니다.');continue
        # Select only from development folds; final test values never participate.
        _,candidate_id,params,record=min(candidates,key=lambda t:(t[0],t[1]))
        try:
            fitted[name]=fit_checked(clone(template).set_params(**params),X,y)
            record['선택됨']=True
            chosen.append({'모델':name,'선택 기준':objective,'선택 설정':record['설정'],
                           '검증 평균 MAE':record['검증 평균 MAE'],'검증 평균 MSE':record['검증 평균 MSE'],
                           '성공 후보 수':len(candidates),'전체 후보 수':len(list(ParameterGrid(SEARCH_SPACES[name])))})
        except Exception as exc:
            errors.append(f'{name}: 선택한 설정의 전체 학습에 실패하여 제외했습니다. ({type(exc).__name__})')
    fitted={'베이스라인':DummyRegressor(strategy='mean').fit(X,y),**fitted}
    scores=[];predictions=test[['repo','month','commits']].rename(columns={'commits':'실제 커밋 수'}).copy()
    for name,model in fitted.items():
        try:
            p=model.predict(test[FEATURES])
            if not np.isfinite(p).all():raise ValueError('유효하지 않은 예측값')
            scores.append({'모델':name,'MAE':mean_absolute_error(test.commits,p),'MSE':mean_squared_error(test.commits,p),
                           'R²':r2_score(test.commits,p) if test.commits.nunique()>1 else np.nan})
            predictions[name]=p
        except Exception as exc:errors.append(f'{name}: 최종 예측에 실패했습니다. ({type(exc).__name__})')
    info={'train':len(train),'test':len(test),'cut':cut,'first':months[0],'last':months[-1],
          'mean':float(train.commits.mean()),'eligible':len(df),'objective':objective,
          'trials':trials,'chosen':chosen,'folds':fold_info,'total_candidates':total}
    return pd.DataFrame(scores),predictions,info,errors
