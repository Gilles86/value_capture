library(lme4)
library(lmerTest)
library(dplyr)
library(ggplot2)

food_data <- read.csv("~/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/Jupyter/food/preprocessed.csv")

food_data = food_data %>% filter(RT_filter == 'True')

food_data$correct <- as.factor(food_data$correct)
food_data$correct <- relevel(food_data$correct, ref = '1')
food_data$dist <- as.factor(food_data$dist)
food_data$dist <- relevel(food_data$dist, ref = 'nd')
food_data$food_rating <- as.factor(food_data$food_rating)

model_v1 <- lmer(RT ~ correct + target_present + dist  + (1 | subject_nr), data = food_data)
summary(model_v1)
fixef(model_v1)

food_dist <- subset(food_data, dist != 'nd')
                               
model_v2 <- lmer(logRT ~ dist + food_rating  + (1 | subject_nr), data = food_dist)
summary(model_v2)

model_v3 <- lmer(RT ~ distractor_id + (1 | subject_nr) , data = food_dist_fake)
summary(model_v3)
fixef(model_v3)


food_dist_real <- subset(food_data, dist == 'real')

model_v3 <- lmer(RT ~ food_rating + (1 | subject_nr) + (1 | distractor_id), data = food_dist_real)
summary(model_v3)

food_dist_fake <- subset(food_data, dist == 'fake')

model_v3 <- lmer(RT ~ distractor_id + (1 | subject_nr) , data = food_dist_fake)
summary(model_v3)
