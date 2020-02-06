##### Multiple seasonal decomposition #####

##### decompose a time series into seasonal, trend and remainder components - seasonal components are estimated 
##### iteratively using STL - multiple seasonal periods are allowed - the trend component is computed for the last 
##### iteration of STL - non-seasonal time series are decomposed into trend and remainder only - in this case, supsmu 
##### is used to estimate the trend - optionally, the time series may be Box-Cox transformed before decomposition - unlike 
##### stl, mstl is completely automated

library(tidyverse)
library(forecast)


mstl(taylor) %>% autoplot()

mstl(AirPassengers, lambda='auto') %>% autoplot()
