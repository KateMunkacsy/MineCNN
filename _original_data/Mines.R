library("readxl")
library(dplyr)
library(tidyr)
library(stringr)



setwd("~/Desktop/mines/data/")
mrdata <- read.csv(file = 'mrds.csv')
deposit_data <- read.csv(file = 'deposit.csv')
diamond_data <- read_excel('diamond.xls', sheet = 'DIADATA')
african_data <- read.csv('africanmines.csv')
geoparsing_data <- read.csv('TRACARMines.csv')

View(mrdata)
dim(mrdata)
mrdata %>% 
  filter(!grepl('Water|Unknown', work_type)) %>% 
  dim()
  View()
mr <- mrdata %>%  
  select(country, latitude, longitude, commod1)%>% 
  rename(commodity = commod1)%>%
  mutate(source = 'mrdata')
mr <- mr[!(mr$commodity == ""),]
head(mr, 4)
dim(mr)
View(mr)








diamond <- diamond_data %>%
  select(COUNTRY, LAT, LONG) %>% 
  rename(country = COUNTRY,
         latitude = LAT,
         longitude = LONG) %>% 
  mutate(commodity = 'diamond', source = 'diamond')
head(diamond, 4)
dim(diamond)


deposit <- deposit_data %>% 
  select(country, latitude, longitude, commodity) %>% 
  mutate(source = 'deposit')
head(deposit, 4)
dim(deposit)
View(deposit)

africa <- african_data %>% 
  rename(location = Mine.Location) %>%
  separate(location, into = c('latitude', 'longitude'), sep = ',') %>% 
  drop_na(latitude, longitude) %>% 
  select(Country, latitude, longitude, Commodity) %>%
  rename(country = Country,
         commodity = Commodity) %>% 
  mutate(source = 'africa')
africa$latitude <- gsub("[() ]", "", africa$latitude)
africa$longitude <- gsub("[() ]", "", africa$longitude)

head(africa, 4)

geo <- geoparsing_data %>% 
  select(country, latitude, longitude, commodity) %>% 
  mutate(source = 'geoParsing')
head(geo, 4)



# merge all of the files
mine_data <- rbind(mr, deposit, diamond, africa, geo)
mine_data %>% 
  dim()

mine_complete <- mine_data[complete.cases(mine_data$latitude, mine_data$longitude),]
dim(mine_complete)

options(digits = 7)
mine_complete$latitude <- mine_complete$latitude %>% 
  as.numeric()
mine_complete$longitude <- mine_complete$longitude %>% 
  as.numeric()
mine_complete$source <- mine_complete$source %>% 
  as.factor()

mine_complete$commodity <- mine_complete$commodity %>% 
  word(sep = ', ') %>% 
  word(sep = '-') %>% 
  tolower() %>% 
  as.factor()
mine_complete <- mine_complete %>% 
  group_by(commodity) %>% 
  filter(n() >= 150)

str(mine_complete)
View(mine_complete)


dim(mine_complete)
final_mine <- distinct(mine_complete)
dim(final_mine)
head(final_mine)
write.csv(final_mine, 'mines.csv')


