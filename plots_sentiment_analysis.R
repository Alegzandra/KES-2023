setwd("C:\\Users\\ciobo\\Desktop\\Ale")

library(dplyr)
library(ggplot2)
library(lubridate)
library(stringr)
library(tsibble)
library(COVID19)
library(wesanderson)
options(scipen=999)
textd<-read.csv("./inferenced_concat_all.csv", header = T, na.strings = "", stringsAsFactors = F,encoding = "UTF-8")
head(textd)
str(textd)
covid_ro<-covid19(country="Romania", start = "2021-01-01", end="2022-02-28")
covid_sel<-covid_ro[,c("date","deaths",'vaccines')]
covid_sel$vaccines<-as.numeric(covid_sel$vaccines)


for(i in c(1:423)){
   
    covid_sel$d_deaths[1+i]<- covid_sel$deaths[1+i]-covid_sel$deaths[i]
    covid_sel$d_vacines[1+i]<-covid_sel$vaccines[1+i]-covid_sel$vaccines[i]
}
covid_sel$date<-as.Date(as.character(covid_sel$date),"%Y-%m-%d")
covid_sel$Year<-year(covid_sel$date)
covid_sel$Month<-month(covid_sel$date)
covid_sel$Week_of_year<-week(covid_sel$date)
covid_sel$year_week<-yearweek(covid_sel$date,week_start = getOption("lubridate.week.start", 1))
 
covid_week<-covid_sel%>%group_by(year_week)%>%summarise(
    wdeaths=sum(d_deaths, na.rm=T),
    wvaccinated=sum(d_vacines,na.rm=T))

textd$Date<-as.Date(as.character(textd$Date), format="%m/%d/%Y")
textd$Week_of_year<-week(textd$Date)
textd$Day_of_year<-yday(textd$Date)
textd$year_week<-yearweek(textd$Date,week_start = getOption("lubridate.week.start", 1))
strptime(textd$Date, format = "%Y-%U")
#textd<-textd[-which(textd$probability<0.6),]


textd$inference.1<-factor(textd$inference.1, levels = c("positive","neutru","negative"),
                          labels = c("Positive","Neutral", "Negative"))

tw_week<-textd%>%group_by(year_week,inference.1)%>%summarise(tweets=n())


df_tw_cov<-merge(tw_week,covid_week,by="year_week",no.dups = T )
#relation between tweets and the number of deaths per week 
cor.test(df_tw_cov[df_tw_cov$inference.1=="Negative","tweets"],df_tw_cov[df_tw_cov$inference.1=="Negative","wdeaths"] )
cor.test(df_tw_cov[df_tw_cov$inference.1=="Positive","tweets"],df_tw_cov[df_tw_cov$inference.1=="Positive","wdeaths"] )
cor.test(df_tw_cov[df_tw_cov$inference.1=="Neutral","tweets"],df_tw_cov[df_tw_cov$inference.1=="Neutral","wdeaths"] )

cor.test(df_tw_cov[df_tw_cov$inference.1=="Negative","tweets"][1:25],df_tw_cov[df_tw_cov$inference.1=="Negative","wdeaths"][1:25])
cor.test(df_tw_cov[df_tw_cov$inference.1=="Positive","tweets"][1:20],df_tw_cov[df_tw_cov$inference.1=="Positive","wdeaths"][1:20] )
cor.test(df_tw_cov[df_tw_cov$inference.1=="Neutral","tweets"],df_tw_cov[df_tw_cov$inference.1=="Neutral","wdeaths"] )



#relation between tweets and the number of persons vaccinated per week 
cor.test(df_tw_cov[df_tw_cov$inference.1=="Negative","tweets"],df_tw_cov[df_tw_cov$inference.1=="Negative","wvaccinated"] )
cor.test(df_tw_cov[df_tw_cov$inference.1=="Positive","tweets"],df_tw_cov[df_tw_cov$inference.1=="Positive","wvaccinated"] )
cor.test(df_tw_cov[df_tw_cov$inference.1=="Neutral","tweets"],df_tw_cov[df_tw_cov$inference.1=="Neutral","wvaccinated"] )

coeff<-0.1

#Tweets vs deaths and their emotions

ggplot(df_tw_cov, aes(x=year_week))+
    geom_bar( stat="identity", position="dodge",aes(y=tweets*10,fill=inference.1))+
    geom_line(aes(y=wdeaths, colour ="No. of Deaths"),size=1, color="darkblue")+
    #scale_x_continuous(breaks = seq(yearweek(textd$Date[1],week_start = getOption("lubridate.week.start", 1)),
    #                                         yearweek( textd$Date[18319],week_start = getOption("lubridate.week.start", 1)),10))+
    scale_y_continuous( name = "Tweets per week",
                        breaks = seq(0,5500,1000),limits = c(0,5500),
                        labels =c("0","100","200","300","400", "500"), 
    sec.axis = sec_axis( ~., name="Deaths"))+
    theme_classic() +
    scale_color_manual(labels = c("Deaths"), values = c("darkblue")) +
    scale_fill_manual(labels = c("Positive", "Netrual", "Negative"), values =wes_palette("Darjeeling1", n=3) ) +
    theme(
        axis.title.y = element_text(color ="gray20", size=10, face="bold"),
        axis.title.x = element_text(color ="gray20", size=10, face="bold"),
        axis.title.y.right = element_text(color = "gray20", size=10,face = "bold"),
        title = element_text(color = "gray20", size=11,face = "bold"),
        legend.position="bottom") +
    guides(fill=guide_legend(title="Tweets emotion"),
           colour=guide_legend(title="Deaths"))+
    xlab("Week of the Year")+
    ggtitle("Weekly Tweets vs. No. Weekly Deaths")
    
    

scale_fill_brewer(palette = "Set2")
    
#all tweets graph ---
ggplot(df_tw_cov )+
    geom_col(aes(x=inference.1,y=tweets, fill=inference.1))+
    scale_y_continuous( name = "Tweets per week",
                        breaks = seq(0,10000,2500),limits = c(0,10000))+
    theme_classic() +
    scale_fill_manual(labels = c("Positive", "Netrual", "Negative"), values =wes_palette("Darjeeling1", n=3) ) +
    theme(
        axis.title.y = element_text(color ="gray20", size=10, face="bold"),
        axis.title.x = element_text(color ="gray20", size=10, face="bold"),
        title = element_text(color = "gray20", size=11,face = "bold"),
        legend.position="bottom") +
    guides(fill=guide_legend(title="Tweets"))+
    xlab("Emotion")+
    ggtitle("Vaccine Related Emotions on Twitter (Jan. 2021 and Feb. 2022)")
    
    
#vaccinated people per week ----    
ggplot(df_tw_cov, aes(x=year_week))+
    geom_bar( stat="identity", position="dodge",aes(y=wvaccinated, fill="Vaccinated people per week"))+
    geom_line(aes(y=wdeaths*100),size=1, color="darkblue")+
    #scale_x_continuous(breaks = seq(yearweek(textd$Date[1],week_start = getOption("lubridate.week.start", 1)),
    #                                         yearweek( textd$Date[18319],week_start = getOption("lubridate.week.start", 1)),10))+
    scale_y_continuous( name = "Vaccinated people per week",
                        breaks = seq(0,850000,100000),limits = c(0,830000),
                        sec.axis = sec_axis( ~., 
                                   name = "Deaths per week",
                                  breaks = seq(0,310000,50000),
                        labels =c("0","500","1000","1500","2000", "2500","3000")))+
    theme_classic() +

    scale_color_manual(name = "", values = c("Deaths per week" = "darkblue"))+
    scale_fill_manual(labels = c("Vaccinated people per week"), values ="wheat3" ) +
    theme(
        axis.title.y = element_text(color ="gray20", size=10, face="bold"),
        axis.title.x = element_text(color ="gray20", size=10, face="bold"),
        axis.title.y.right = element_text(color = "gray20", size=10,face = "bold"),
        title = element_text(color = "gray20", size=11,face = "bold"),
        legend.position="bottom",legend.spacing.y = unit(-0.2, "cm")) +
    guides(fill=guide_legend(title=""),
           colour=guide_legend(title=""))+
    xlab("Week of the Year")+
    ggtitle("Weekly Vaccinated vs. No. of Deaths")




#vaccinated people per week ----    
ggplot(df_tw_cov, aes(x=year_week))+
    geom_bar( stat="identity", position="dodge",aes(y=wvaccinated, fill="Vaccinated people per week"))+
   # geom_line(aes(y=wdeaths*100),size=1, color="darkblue")+
    #scale_x_continuous(breaks = seq(yearweek(textd$Date[1],week_start = getOption("lubridate.week.start", 1)),
    #                                         yearweek( textd$Date[18319],week_start = getOption("lubridate.week.start", 1)),10))+
    scale_y_continuous( name = "Vaccinated people per week",
                        breaks = seq(0,850000,100000),limits = c(0,830000))+
                        #sec.axis = sec_axis( ~., 
                        #                     name = "Deaths per week",
                         #                    breaks = seq(0,310000,50000),
                         #                    labels =c("0","500","1000","1500","2000", "2500","3000")))+
    theme_classic() +
    # scale_color_manual(name = "", values = c("Deaths per week" = "darkblue"))+
    scale_fill_manual(labels = c("Vaccinated people per week"), values ="wheat3" ) +
    theme(
        axis.title.y = element_text(color ="gray20", size=10, face="bold"),
        axis.title.x = element_text(color ="gray20", size=10, face="bold"),
        axis.title.y.right = element_text(color = "gray20", size=10,face = "bold"),
        title = element_text(color = "gray20", size=11,face = "bold"),
        legend.position="bottom",legend.spacing.y = unit(-0.2, "cm")) +
    guides(fill=guide_legend(title=""),
           colour=guide_legend(title=""))+
    xlab("Week of the Year")+
    ggtitle("Weekly Vaccinated vs. No. of Deaths")

    

ggplot(textd)+
    geom_boxplot( aes(x=inference.1,y=probability,colour=inference.1, shape=inference.1))+
    theme_classic()
ggplot(textd)+
    geom_violin( aes(x=inference.1,y=probability,fill=inference.1, shape=inference.1))+
    theme_classic()
head(tw_week)
