from transformers import pipeline

summarizer = pipeline("summarization", model="stevhliu/my_awesome_billsum_model")

translator = pipeline("translation", model="facebook/nllb-200-distilled-1.3B", src_lang='eng_Latn', tgt_lang='kor_Hang', max_length=512)

news = """ 
Chinese leader Xi Jinping on Thursday urged regional leaders to resist “external interference” at a gathering of a Eurasian security bloc touted by Beijing and Moscow as a counterbalance to Western power.

Addressing the Shanghai Cooperation Organization (SCO)’s annual leaders’ summit in Kazakhstan, Xi called on member states to “consolidate the power of unity” in the face of “the real challenge of interference and division.”

“We should work together to resist external interference … and firmly grasp our own future and destiny, as well as regional peace and development, in our own hands,” Xi was quoted as saying by Chinese state broadcaster CCTV.

The 10-member bloc must handle internal differences with peace, seek common ground, and resolve difficulties in cooperation, Xi added.

Founded in 2001 by China, Russia, Kazakhstan, Kyrgyzstan, Tajikistan and Uzbekistan to combat terrorism and promote border security, the SCO has grown in recent years as Beijing and Moscow drive a transformation of the bloc from a regional security club with a focus on Central Asia to a geopolitical counterweight to Western institutions led by the United States and its allies. """

# 'summary': "신작 부재에도 매출 18.3% 증가 전망…전체 사업 40% 차지\n반도체 깜짝 실적에 영업익 비중은 50%→30%로 감소",
#  'text' : "삼성전자의 스마트폰을 담당하는 MX(모바일경험) 사업부가 신작 부재에도 AI(인공지능)폰의 힘으로 실적 방어에 성공했다. 증권가 예측이 다소 엇갈리지만, 최소 지난해 2분기보다는 개선됐을 것으로 추정된다.\n삼성전자는 5일 2024년도 2분기 잠정 실적발표에서 매출액과 영업이익이 각각 74조원, 10조4000억원을 기록했다고 밝혔다. 전년 동기 대비 각각 23.3%, 1452.24% 증가한 수치다. 반도체 부문이 기대 이상의 성과를 거두면서 영업이익은 시장 컨센서스를 2조원 이상 웃돌았다. 1분기에 이어 두 분기 연속 어닝서프라이즈다.\n이날 발표는 잠정치라 사업 부문별 실적은 공개되지 않는다. 다만 국내 증권사들의 최근 전망치를 고려하면 삼성전자의 MX사업부 2분기 매출액은 26조1790억~30조2370억원 사이로 예상된다. 전년 동기 대비 2.4~18.3% 증가한 수치다. 영업이익은 2조5000억~3조3260억원으로 지난해 2분기와 비슷하거나 최대 9.4% 증가했을 전망이다.\n신작 출시 효과가 사라지며 지난 1분기보다는 매출과 영업이익이 모두 감소했으나, 업계는 최초 AI폰인 갤럭시 S24가 기대 이상의 판매량을 기록하고 있다고 내다봤다. 다만 스마트폰에 탑재되는 AP(애플리케이션 프로세서) 등 부품 원가가 상승하면서 영업이익이 다소 아쉽다는 평가도 나온다.\n김선우 메리츠증권 연구원은 '스마트폰과 태블릿 출하량이 각각 5300만대, 700만대를 기록했지만, 갤럭시 S24 시리즈 판매량이 810만대로 감소했고 메모리 제품 원가 상승에 따라 이익률 하락이 발생했을 것'이라고 추정했다. 박유악 키움증권 연구원도 '스마트폰의 판매량은 예상치에 부합하지만, 메모리 가격 인상에 따른 수익성 둔화가 나타날 것으로 판단된다'고 설명했다.\n 증권가는 3분기 삼성전자 실적이 회복될 것으로 내다봤다. 오는 10일 갤럭시 Z폴드·플립6 언팩이 프랑스 파리에서 열리기 때문이다. 아울러 갤럭시 워치 울트라·갤럭시링 등 새 제품군도 선보인다. 약 3년 만에 신제품이 나오는 갤럭시 버즈3 시리즈도 에어팟과 같은 '콩나물' 형태로 디자인이 대폭 바뀌면서 기대감을 모은다. 상상인증권은 오는 3분기 삼성전자 MX사업부의 매출액이 35조1370억원, 영업이익이 4조3220억원으로 대폭 개선될 것이라고 예상했다.\n한편, 삼성전자는 오는 31일 2분기 컨퍼런스 콜을 진행할 예정이다.",
#  'title': "삼성전자 MX사업부, AI폰으로 2분기 연속 실적 지켰다"

summary = summarizer(news, max_length=50)
translation = translator(str(summary))

print(summary)
print(translation)