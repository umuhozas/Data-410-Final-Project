#### Data-410-Final-Project
#### Solange Umuhoza
#### 11 May 2022

# How Happy Are We Actually?

### Introduction

The goal of this project is to explain and predict happiness around the world. This project will use a "World Happiness Report up to 2022" dataset found on Kaggle.com and it contains data from 2015 up to 2022. the datasets used contain information from the Gallup World Survey. This project will focus on the data from 2019 and 2020. I am interested in comparing how the Covid-19 pandemic affected the overall happiness around the world. Each country has a happiness score that is achieved by considering information collected from survey-based research where respondents responded to main life evaluation questions and ranked their current lives on a scale of zero to ten. In every country, a sample size of 2000 to 3000 people was asked to think of a ladder where the best possible life for them would be a ten and the worst possible experience being a zero. Survey respondents answered questions like “If you were in trouble, do you have relatives or friends you can count on to help you whenever you need them?” or “Have you donated money to a charity in the past month?”, or “Are you satisfied by or dissatisfied with your freedom to choose what to do with your life?” These questions helped in determining the level of social support, generosity, and freedom that citizens of different countries have.

### Description of the data 

Each variable measured reveals a populated-weighted average score on a scale running from 0 to 10 that is tracked over time and compared against other countries. These variables currently include real GDP per capita, social support, healthy life expectancy, freedom to make life choices, generosity, and perceptions of corruption. 

GDP: GDP per capita is a measure of the overall domestic production and it functions as a comprehensive scorecard of a given country’s economic health. 

social: Social support means the ability to have family and friends or other people who can help you in time of need. Social support improves happiness in people’s lives because they do not have to worry about being alone in difficult situations.

health: Healthy Life Expectancy is the average years of life in good health. Life without disability illnesses or injuries.

freedom: Freedom of choice describes an individual’s ability to choose what they do with their life. The average of all answers determined the result of every country.

generosity: Respondents were asked whether they have donated money to a charity in the past month. The average of all answers determined the result of every country.

corruption: The Corruption Perceptions Index (CPI) is an index published annually by Transparency International since 1995, which ranks countries “by their perceived levels of public sector corruption, as determined by expert assessments and opinion surveys.”

In some cases where countries are missing one or more happiness factors over the survey period, information from earlier years is used as if they were current information. This may cause some bias in my results but it will not make a huge difference because there is a limit of 3 years for how far back the researchers went in search of those missing values. I believe that the dataset used in this project is good, but not 100% accurate respective to years.

### Data Cleaning
While cleaning the datasets used in this project, I wanted to maintain variables that play an important role in explaining national happiness. For both 2019 and 2020, I kept log GDP per capita, Social support, Healthy life expectancy, Freedom to make life choices, Generosity, and Perception of corruption. Luckily, I did not have any missing values for both datasets.

![data](https://user-images.githubusercontent.com/98835048/167772309-06e657b9-4a7c-40e0-bd38-4a62244dcb67.png)


### Correlation Coefficients for all Numerical variables
I started my analysis by checking the correlation coefficients between all variables to have an idea of what correlates more with the happiness score.  GDP, social support,  and health expectancy have the highest correlation coefficient with the happiness score for both 2019 and 2020. Correlations are useful to get a quick idea about the data.


![heatmap_2019 (1)](https://user-images.githubusercontent.com/98835048/167767733-32c328d2-0640-49ae-9bc5-0a96a44d83fc.png)
![heatmap_2020 (1)](https://user-images.githubusercontent.com/98835048/167767731-0308d4af-1a82-45a2-9915-c47a703bd0bf.png)

After obtaining the correlation coefficients for all numerical values in my dataset, I used Tableau to obtain correlation visualizations to see how all independent features relate to our dependent feature.


<img width="500" alt="gdp health" src="https://user-images.githubusercontent.com/98835048/167769134-b745261d-591d-4040-b5e2-f911f88cca9f.png">
<img width="500" alt="Social freedom" src="https://user-images.githubusercontent.com/98835048/167769144-1e9a8d2f-6cc0-4337-af67-cb5fa880c57a.png">
<img width="500" alt="generosity corrupt" src="https://user-images.githubusercontent.com/98835048/167769151-afda2df4-72b5-4f67-977b-e21f98afb2a8.png">

### Description of all methods applied










