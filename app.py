from pathlib import Path
import io
import hashlib
import streamlit as st
import pandas as pd
import numpy as np
from analysis import local_files, github_files, request, aggregate, evaluate

def rebuild_scores(scores, predictions):
    """Normalize old/new result schemas from the same observed predictions."""
    if not isinstance(scores, pd.DataFrame) or not isinstance(predictions, pd.DataFrame):
        raise ValueError('이전 학습 결과를 읽을 수 없습니다. 다시 학습해 주세요.')
    if '모델' not in scores or '실제 커밋 수' not in predictions:
        raise ValueError('점수 계산에 필요한 결과가 없습니다. 다시 학습해 주세요.')
    actual = pd.to_numeric(predictions['실제 커밋 수'], errors='coerce').to_numpy(dtype=float)
    if not len(actual) or not np.isfinite(actual).all():
        raise ValueError('실제 커밋 수가 올바르지 않습니다. 데이터를 확인하고 다시 학습해 주세요.')
    rows = []
    variance = np.sum((actual - actual.mean()) ** 2)
    for name in scores['모델']:
        if name not in predictions:
            raise ValueError(f'{name}의 예측 결과가 없습니다. 다시 학습해 주세요.')
        predicted = pd.to_numeric(predictions[name], errors='coerce').to_numpy(dtype=float)
        if not np.isfinite(predicted).all():
            raise ValueError(f'{name}의 예측값이 올바르지 않습니다. 다시 학습해 주세요.')
        error = predicted - actual
        rows.append({'모델': name, 'MAE': float(np.mean(np.abs(error))),
                     'MSE': float(np.mean(error ** 2)),
                     'R²': float(1 - np.sum(error ** 2) / variance) if variance else np.nan})
    result = pd.DataFrame(rows, columns=['모델', 'MAE', 'MSE', 'R²'])
    if result.empty or '베이스라인' not in result['모델'].values:
        raise ValueError('베이스라인 결과가 없습니다. 다시 학습해 주세요.')
    return result

st.set_page_config(page_title='yulstar · AI 코딩 탐구실',page_icon='📊',layout='wide')
st.markdown('''<style>.stApp{background:#f6f8fa}h1,h2,h3{color:#173d39}.stButton>button[kind="primary"]{background:#087e72;border:0}div[data-testid="stMetric"]{background:white;padding:18px;border:1px solid #dce6e5;border-radius:12px}div[data-testid="stMetricValue"]{color:#087e72}main .block-container{max-width:1200px;padding-top:2.5rem}</style>''',unsafe_allow_html=True)
st.caption('YULSTAR / AI 코딩 탐구실')
st.title('AI를 많이 쓰면, 개발 활동도 늘어날까?')
st.write('깃허브의 실제 커밋 기록으로 다음 달 커밋 수를 예측하고, 모델의 점수를 비교해요.')
st.caption('커밋 = 코드를 수정하고 저장한 기록. 이 앱은 코딩의 양을 살펴보며, 코드의 질이나 AI의 인과적 효과를 직접 측정하지 않아요.')

@st.cache_data(show_spinner=False)
def load_local(signatures):
    return aggregate((p,Path(p).read_bytes()) for p,_,_ in signatures)
@st.cache_data(show_spinner=False,ttl=3600)
def list_remote(url,branch):return github_files(url,branch)
@st.cache_data(show_spinner=False,ttl=3600)
def load_remote(selected):return aggregate((name,request(url)) for name,url in selected)

st.header('01  데이터 준비')
mode=st.radio('데이터가 어디 있나요?',['앱과 같은 깃허브 저장소','다른 공개 깃허브 저장소','직접 파일 올리기'],horizontal=True)
ready=None;selection_key=None
if mode=='앱과 같은 깃허브 저장소':
    root=Path(__file__).resolve().parent
    files=local_files(root);choices={str(p.relative_to(root)):p for p in files}
    if not files:st.info('앱과 같은 폴더 또는 그 아래 폴더에 CSV·ZIP을 넣어 주세요. 데이터가 다른 저장소에 있으면 위에서 두 번째 방법을 고르세요.')
    else:
        preferred=[n for n in choices if 'commit' in n.lower() or n.lower().endswith('.zip')]
        selected=st.multiselect('읽을 파일 (연도별 파일을 모두 선택하세요)',list(choices),default=preferred or list(choices))
        signatures=tuple((str(choices[n]),choices[n].stat().st_size,choices[n].stat().st_mtime_ns) for n in selected)
        selection_key=('local',signatures)
        st.caption('저장소에 이미 내려받아진 파일을 직접 읽으므로 GitHub API 사용 한도와 관계없어요. ZIP 안의 CSV도 전부 읽어요.')
        if st.button('선택한 데이터 읽기',type='primary',disabled=not selected):
            try:
                with st.spinner('전체 행을 읽고 월별로 정리하는 중입니다. 큰 파일은 몇 분 걸릴 수 있어요.'):
                    ready=load_local(signatures)
            except Exception as e:st.error(str(e))
elif mode=='다른 공개 깃허브 저장소':
    url=st.text_input('공개 저장소 주소',placeholder='https://github.com/사용자/저장소')
    branch=st.text_input('브랜치 (비워 두면 기본 브랜치)',placeholder='main')
    if st.button('깃허브 파일 목록 확인',disabled=not url):
        try:
            with st.spinner('파일 목록 확인 중…'):st.session_state.remote=(url,branch,list_remote(url,branch))
        except Exception as e:st.error(str(e))
    remote=st.session_state.get('remote')
    if remote and remote[:2]==(url,branch):
        options=dict(remote[2]);names=list(options)
        if not names:st.warning('CSV 또는 ZIP 파일이 없습니다.')
        selected=st.multiselect('가져올 데이터',names,default=[n for n in names if 'commit' in n.lower() or n.endswith('.zip')])
        chosen=tuple((n,options[n]) for n in selected);selection_key=('remote',url,branch,chosen)
        st.caption('공개 저장소만 지원해요. 접속이나 API 한도 오류가 나면 데이터를 앱과 같은 저장소에 넣어 주세요. 데이터 내용 변경 시 아래 캐시 초기화를 눌러 주세요.')
        if st.button('선택한 데이터 가져오기',type='primary',disabled=not selected):
            try:
                with st.spinner('깃허브에서 데이터를 내려받고 월별로 정리하는 중…'):ready=load_remote(chosen)
            except Exception as e:st.error(str(e))
else:
    uploads=st.file_uploader('CSV·ZIP 여러 개를 선택하세요',type=['csv','zip'],accept_multiple_files=True)
    selection_key=('upload',tuple((u.name,u.size,u.file_id) for u in uploads))
    if st.button('올린 데이터 읽기',type='primary',disabled=not uploads):
        try:
            with st.spinner('전체 행을 읽는 중…'):ready=aggregate((u.name,u.getvalue()) for u in uploads)
        except Exception as e:st.error(str(e))

if st.button('파일 읽기 캐시 초기화'):
    st.cache_data.clear();st.session_state.pop('loaded',None);st.session_state.pop('result',None);st.session_state.pop('remote',None);st.rerun()
if ready is not None:
    st.session_state.loaded=(selection_key,ready);st.session_state.pop('result',None)
loaded=st.session_state.get('loaded')
if not loaded or selection_key!=loaded[0]:
    st.info('데이터를 선택하고 읽기 버튼을 눌러 주세요. 선택을 바꾸면 데이터를 다시 읽어야 해요.')
    st.stop()
monthly,counts,overview,used,skipped=loaded[1]
a,b,c,d=st.columns(4)
a.metric('읽은 커밋 행',f'{counts["total"]:,}');b.metric('원본 열',len(overview));c.metric('저장소',monthly.repo.nunique());d.metric('저장소별 월 기록',f'{len(monthly):,}')
with st.expander('빈 값과 데이터 처리 과정 보기'):
    st.write(f'처리 전 **{counts["total"]:,}행** → 처리 후 **{counts["retained"]:,}행**')
    st.write(f'날짜·저장소 오류 {counts["invalid"]:,}행, 중복 {counts["duplicates"]:,}행, 일반 자동화 봇 {counts["bots"]:,}행 제외.')
    st.write(f'알 수 없는 AI 표시 {counts["unknown_ai"]:,}개는 AI 비율의 분모에서 제외. 빈 작성자 ID {counts["missing_author"]:,}개는 개발자 수에서 제외.')
    if counts['no_hash']:st.warning(f'커밋 해시가 없는 {counts["no_hash"]:,}행은 중복 여부를 확인하지 못했어요.')
    st.write('AI 에이전트의 커밋은 포함해요. 큰 수치도 실제 프로젝트 활동일 수 있어서 이상치를 자동으로 삭제하지 않아요. 기록이 없는 달은 0으로 채우지 않아요.')
    st.dataframe(overview,hide_index=True,use_container_width=True)
    st.write('읽은 파일:',used)
    if skipped:st.caption('필수 커밋 열이 없어 건너뛴 파일: '+', '.join(skipped))

st.header('02  실험 설정')
months=sorted(monthly.month.unique());initial=next((i for i,m in enumerate(months) if m>='2023-01'),0)
a,b,c=st.columns(3)
start=a.selectbox('시작 월',months,index=initial)
end=b.selectbox('마지막 월',months,index=max(0,len(months)-2) if len(months)>1 else 0)
ratio=c.selectbox('학습에 쓸 기간',[.8,.7,.9],format_func=lambda v:f'앞 {v:.0%} 학습 / 뒤 {1-v:.0%} 평가')
st.caption('마지막 달은 수집이 덜 끝났을 수 있어 기본적으로 제외해요. 그 달이 완전한 데이터라면 직접 포함할 수 있어요.')
st.write('**입력:** 지난달 AI 표시 비율·개발자 수·커밋 수·시간 흐름 → **예측:** 이번 달 커밋 수')
st.caption('같은 달의 저장소들은 모두 같은 학습/평가 구간에 들어가요. 바로 전월 기록이 없거나 AI 비율을 계산할 수 없는 행은 제외해요. 평가할 때도 매달 실제 지난달 기록을 사용해요.')
settings=('mse-mae-v2',selection_key,start,end,ratio)
if st.button('모델 학습하고 점수 비교하기',type='primary'):
    if start>end:st.error('시작 월이 마지막 월보다 늦어요.')
    else:
        try:
            with st.spinner('같은 데이터로 네 가지 모델을 학습하고 있어요…'):
                st.session_state.result=(settings,evaluate(monthly,start,end,ratio))
        except Exception as e:st.error(str(e))
result=st.session_state.get('result')
if not result or result[0]!=settings:
    st.info('설정을 정한 뒤 학습 버튼을 누르면 실제 점수가 나와요. 설정 변경 후에는 다시 학습해야 해요.');st.stop()
scores,pred,info,errors=result[1]
try:
    scores=rebuild_scores(scores,pred)
except ValueError as exc:
    st.session_state.pop('result',None)
    st.error(str(exc))
    st.stop()
for e in errors:st.warning(e)
st.header('03  점수 비교')
st.write(f'학습: {info["first"]} ~ {str(pd.Period(info["cut"],freq="M")-1)} / {info["train"]:,}행 · 평가: {info["cut"]} ~ {info["last"]} / {info["test"]:,}행')
st.info('MAE: 평균적으로 커밋 수를 몇 개 틀렸는지예요. MSE: 오차를 제곱한 뒤 평균을 내요. 크게 틀린 예측에 더 큰 벌점을 줘요. 둘 다 낮을수록 좋아요. 정확도 %가 아니에요.')
st.dataframe(scores.style.format({'MAE':'{:.2f}','MSE':'{:.2f}','R²':'{:.3f}'},na_rep='계산 불가'),hide_index=True,use_container_width=True)
best=scores.loc[scores.MAE.idxmin()];base=scores[scores['모델']=='베이스라인'].iloc[0]
a,b=st.columns(2)
with a:
    st.subheader('모델별 오차 비교')
    metric=st.radio('그래프에서 비교할 점수',['MAE','MSE'],horizontal=True,key='score_metric_mse_mae_v2')
    st.caption(f'가로: 모델 / 세로: {metric} (' + ('커밋 개수' if metric=='MAE' else '커밋 개수의 제곱') + ')')
    st.bar_chart(scores.set_index('모델')[[metric]],color='#087e72')
    improvement=(1-best.MAE/base.MAE)*100 if base.MAE else 0
    st.write(f'**{best["모델"]}**의 MAE는 **{best.MAE:.2f}개**로 가장 낮았어요. 베이스라인보다 오차가 **{improvement:.2f}%** 작았어요.' if best['모델']!='베이스라인' else f'이번에는 베이스라인의 MAE **{base.MAE:.2f}개**가 가장 낮았어요.')
with b:
    st.subheader('실제값과 예측값의 변화')
    st.caption('가로: 평가 월 / 세로: 저장소당 평균 커밋 수')
    trend=pred.groupby('month')[['실제 커밋 수',best['모델']]].mean();st.line_chart(trend,color=['#243b40','#087e72'])
    last=trend.iloc[-1];st.write(f'**{trend.index[-1]}** 실제 평균은 **{last.iloc[0]:.2f}개**, {best["모델"]}의 예측 평균은 **{last.iloc[1]:.2f}개**였어요.')
st.subheader('보고서에 붙일 문장')
text=f'우리 팀은 깃허브 커밋 데이터를 이용해 월별 커밋 수를 예측하는 회귀 실험을 했습니다. 학습 데이터의 평균인 {info["mean"]:.2f}개를 항상 예측하는 베이스라인의 MAE는 {base.MAE:.2f}, MSE는 {base.MSE:.2f}였습니다. '
text+=' '.join(f'{r["모델"]}의 MAE는 {r.MAE:.2f}, MSE는 {r.MSE:.2f}였습니다.' for _,r in scores.iterrows() if r['모델']!='베이스라인')
text+=f' 평가 기간은 {info["cut"]}부터 {info["last"]}까지이며, MAE가 가장 낮은 모델은 {best["모델"]}였습니다. 이 점수는 커밋 수 예측 성능을 보여 주며 AI가 개발량 증가의 원인이라는 증거나 코드 품질 향상의 증거는 아닙니다.'
st.text_area('아래 문장을 복사해서 사용하세요',text,height=200)
a,b,c=st.columns(3)
a.download_button('점수 CSV 저장',scores.to_csv(index=False).encode('utf-8-sig'),'model_scores.csv','text/csv')
b.download_button('실제값·예측값 CSV 저장',pred.to_csv(index=False).encode('utf-8-sig'),'predictions.csv','text/csv')
c.download_button('보고서 문장 저장',text.encode('utf-8'),'report.txt','text/plain')
with st.expander('모델과 해석에 대한 설명'):
    st.write('베이스라인은 학습 데이터 평균을 예측해요. 선형 회귀는 여러 입력의 선형 관계를 배워요. 포아송 회귀는 0 이상의 개수를 다루고, 랜덤 포레스트는 여러 나무의 예측을 평균해요.')
    st.write('scikit-learn 모델을 사용해요. 포아송은 alpha=0.01, 최대 반복 3,000회. 랜덤 포레스트는 나무 150개, 깊이 12, 잎 최소 5행, random_state=42예요. 선형 회귀의 음수 예측도 그대로 평가해요. 평균·표준편차는 학습 데이터로만 계산해요.')
    st.write('R²는 평균값만 예측하는 것과 비교한 지표예요. 1에 가까울수록 좋고 음수도 가능해요. 실제값이 모두 같으면 표시하지 않아요. 모델 설정과 구현이 달라 이전 웹 버전과 점수가 다를 수 있어요.')
    st.write('AI 표시가 없는 커밋도 AI를 사용했을 수 있어요. AI 관련 오픈소스 프로젝트만으로 모든 개발자에게 일반화할 수 없어요. 이 실험만으로 코드 품질이나 AI 자체의 효과를 판단하지 않아요.')
st.caption('데이터 출처: Abiram Bharathi S.A · Who Writes AI Code? 1.3M Open-Source AI Commits · CC BY 4.0')
