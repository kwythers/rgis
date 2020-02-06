#### time series  decomposition ##### 

library(tidyverse)
library(fpp2)

# assumes seasonal component repeats over time - not always the case
elecequip %>% decompose(type="multiplicative") %>%
  autoplot() + xlab("Year") +
  ggtitle("Classical multiplicative decomposition
    of electrical equipment index")
##### the run of remainder values below 1 in 2009 suggests that there is some “leakage” of the trend-cycle component 
##### into the remainder component - the trend-cycle estimate has over-smoothed the drop in the data, and the corresponding 
##### remainder values have been affected by the poor trend-cycle estimate

# X11 decomp 
library(seasonal)
elecequip %>% seas(x11 = "") -> fit
autoplot(fit) +
  ggtitle("X11 decomposition of electrical equipment index")
##### the X11 trend-cycle has captured the sudden fall in the data in early 2009 better than either of the methods, 
##### and the unusual observation at the end of 2009 is now more clearly seen in the remainder component

##### trend-cycle component and the seasonally adjusted data, along with the original data
autoplot(elecequip, series = "Data") +
  autolayer(trendcycle(fit), series = "Trend") +
  autolayer(seasadj(fit), series = "Seasonally Adjusted") +
  xlab("Year") + ylab("New orders index") +
  ggtitle("Electrical equipment manufacturing (Euro area)") +
  scale_colour_manual(values=c("gray","blue","red"),
                      breaks=c("Data","Seasonally Adjusted","Trend"))

##### SEATS decomposition ##### 
##### seasonal extraction in ARIMA time series
##### widely used by government agencies around the world
library(seasonal)
elecequip %>% seas() %>%
  autoplot() +
  ggtitle("SEATS decomposition of electrical equipment index")

##### STL decomposition #####
##### a versatile and robust method for decomposing time series - “seasonal and trend decomposition using loess”
#####b several advantages over the classical, SEATS and X11 decomposition methods
 # 1. will handle any type of seasonality
 # 2. the seasonal component is allowed to change over time, and the rate of change can be controlled by the user
 # 3. the smoothness of the trend-cycle can also be controlled by the user
 # 4. can be robust to outliers
 # ***** downside - it does not handle trading day or calendar variation automatically, and it only provides facilities 
 # ***** for additive decompositions
elecequip %>%
  stl(t.window = 13, s.window="periodic", robust = TRUE) %>%
  autoplot()
