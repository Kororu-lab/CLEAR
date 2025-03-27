## Main Objective For NOW
Main Objective of the project will be applying the idea of AiRS with clustering news by NAVER, to non-personalized and following stock impact(delta) too. As Previous prompts, it should be first getting those functionality of NAVER mechanisms, then making our own components with more advanced methods, with stock prices.
Alike first and second tries, I gave you the actual repo zip and base paper, so refer to.

## Revision

- Stock Impact: Need to use of those given stock data, not current style of taking hour of ... etc. It have only the prices from market opens and closes so you'd better design it to make use of them wisely.
- For pos/neg better for using KO-finbert. While keeping the original methods, please extend to more advanced models, and for those experimental ones, implement in jupyter notebook demo file too.
- For those models, I highly recommend for you to find yourself the model which already made and fit our needs, from HuggingFace. Fine-tuning those would be another option too. Any models used and using in system should be explained both in demo and report.
- This time, unlike those other works from you to making actual structure of AiRS, you will be more encouraged to make other tries using various tech. Machine is torch capable, and do not worry about resource, as told. But, you should NOT make any deletion of original AiRS mechanism(for non-personalization). Otherwise, it's all on you, even if you using vllm serve to use actual llm prompt engineering, or fine-tuning, or API embeddings, etc.

## Previous Prompts

I told you to apply idea of NAVER corps’ AiRs and news clustering system, to mimic them and run without personalization, to make complete repo in my Github. Well, following is prompt.

### First Prompt:
Want to delve deep for NAVER corps’ algorithm to provide news recommendation for each users, including clustering and etc. could refer to URL below and my given paper about AiRs system.

https://media.naver.com/algorithm # access and find out ALL of their methods

Well, in final, I want to make similar system, to get the trend and SI(Social Impact), with clustering, to make news in group and recommend wisely, but with more focused on stock impact. For those personalized indexes, please simply ignore them(since there would not be single ‘individuals’ using the service, and this part would be made differently). I made the github repo at: (https://github.com/Kororu-lab/CLEAR) and finally it would be made as repo. So keep on mind with detailed structure of actual repo(recommended to return us a whole repo in folder, to push directly)

To make those system, focus on making similar structure given in those websites’ description and paper. Use the SAME mechanism/algorithm given in, only difference would be made as additional module or if some part cannot directly known. For those data needed, if hard to get by yourself, you could simply get info of those(including each cols, etc.), we will prepare them in real so write on final report.

For each param in papers/each single components of web explanation, make sure all given in your code structure, but personalization part just drop it and mention on report that you dropped.

For the report, made English ver then translate to Korrean too. use your llm to translate not translation modules.


### Revision List
However, I found I didn’t gave you sufficient information. For those actual crawling and stock data already had previous data format so want for you to follow and make the system fit to. There, I gave my actual data in ./data/news and ./data/stock. Check them. Plus, changed some code from before, so check. For your previous works, find out from ./doc. Specifically given below. Finally, It seems that you need full revision for all of codes.

- news_crawler: I gave you the format of our crawled news data, with the colname of {Title,Date,Press,Link,Body,Emotion,Num_comment,AI Summary}, I prefer to use last crawling mechanism, so want for you to follow. Date format will be given as(20250101 18:56), so process from there. THAT will be stored data.
- news crawler: also, for AI_Summary, some of news will have such data, You may feel it optional and make config to use it or not.
- stock_data_collector: We also have the former format of stock data, which only have {Date,Time,Start,High,Low,End,Volume} col, you could use them, may process them but with ONLY given data. If need more, tell me what MUST be given. Well, however, based on the paper given, if not so critical, I prefer you to use minimum info. Delta of each col would be useful info too?
- text_preprocessor: you could consider the article mainly KOREAN, and only a few of them will "contain" english a bit. Different environment from NAVER so would be easier. Focus on Korean data. Also, should make config option to give list of stopwords more advanced, with using advanced tools such as list from Mecab or other tokenizers.
- News_Vectorizer: try making the option to consider title and contents differently, since body context length could vary a lot, should make config option to use which or combined.
- ./src/model: From paper about SI(press) and such part, don't seems to be useless. There, please do not CUT-OFF or MODIFY ANY mechanism unless it is personlization part, and simply add part for financial part(only for those using minimum data; better with using given data). For those all of score(CF, CBF, QE, SI, Latest...) given in paper, NO ignore and make them. You could simply add config whether or not using the data after making.

- For pos/neg: using KR-finbert could be a kind of option. But, not for an actual impact level. This kind of approach seems irreliable for short body so recommended to push for processed/summarized or even title of article.
- For machine environment, it is such a good idea to dynamically get advantages from our gigantic CUDA machine. Recommend to use torch base if using DL, but tf also possible. Would be wonderful to process impact score using DL!

- While trying embedding: could use various methods, including OpenAI API one, so config could have such option, normally FOLLOW THE PAPER given to you
- If making the model keep working, it would be good to update 2 times per day(300/day news would be given at real use, from YNA), at the time those stock market open/closing(maybe using schedule module?), however, you should make the test version of system to just infer once too.

- For those evaluation part from your report, I cannot know how did you made such a scoring from no data, no labels. If you want to implement those evaluations, please try with my ACTUAL data, but remember there’s no clear LABELs there.

## Given Data
I gave you whole repo as one zip file, so you should make revision from here. After the task ends, you should zip them and return me full updated repo. As done before, no translate modules but your llm capabilities.
