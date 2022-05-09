# load the required packages
library(shiny)
require(shinydashboard)
library(ggplot2)
library(dplyr)

# #library(keras)
# install.packages('flexdashboard')
# #install.packages('billboarder')
# library(flexdashboard)
# #library(billboarder)
# library(tidyverse)
# #library(tidyquant)
# #library(corrr)
# library(scales)
# #library(lime)
# library(glue)

#read data and get list of customers
df <- read.csv("data/modelOut.csv", header = TRUE)
custs <- sort(unique(df$custid))

distheight = "80px"

#Dashboard header carrying the title of the dashboard
header <- dashboardHeader(title = "Churn Modeling")  

#Sidebar content of the dashboard
sidebar <- dashboardSidebar(
    #sidebarMenu(
      menuItem("Customer Care Dashboard", tabName = "dashboard", icon = icon("dashboard")
         ,startExpanded = T
         #,selected = T
         ,menuSubItem(selectInput("custID", "Customer ID",
                                 choices = custs,
                                 multiple=FALSE,
                                 width = '98%')))
      ,menuItem("Model Health Dashboard", tabName = "model", icon = icon("cog"))
    #  )
)
    
    
#define content of dashboard page
frow <- fluidRow(
  column( width = 8,
            
          box(
              title = "Likelihood to Churn"
              #,background = "blue"
              ,collapsible = FALSE
              ,solidHeader = FALSE
              ,width = NULL
              ,status = "primary"
              ,plotOutput("churn", height = "200px")
              ),
        
            column(width=6,
                   valueBoxOutput("value1", width=NULL)
                   ),
            column(width=6,
                  valueBoxOutput("value2", width=NULL)
                  )
          ),

  column(width=4, 
             box(
               "Comparisons to Baseline"
               #,status = "primary"
               ,solidHeader = F 
               ,collapsible = F 
               ,width = NULL
               ,height = 35
               #,plotOutput("header", height = "10px")
             )
              ,box(
               title = "% Dropped Calls"
               ,status = "primary"
               ,solidHeader = T 
               ,collapsible = TRUE 
               ,width = NULL
               ,plotOutput("dropperc", height = distheight)
             )
             ,box(
               title = "Mins Used per Month"
               ,status = "primary"
               ,solidHeader = T 
               ,collapsible = TRUE 
               ,width = NULL
               ,plotOutput("mins", height = distheight)
             )
             ,box(
               title = "Consec Months as Cust"
               ,status = "primary"
               ,solidHeader = T 
               ,collapsible = TRUE 
               ,width = NULL
               ,plotOutput("consecmonths", height = distheight)
             )
             ,box(
               title = "Income"
               ,status = "primary"
               ,solidHeader = T 
               ,collapsible = TRUE 
               ,width = NULL
               ,plotOutput("income", height = distheight)
             )
         )
)

#,tags$head(tags$style(HTML(".small-box {height: 50px}")))


body <- dashboardBody(
  #tabItems(
    tabItem(tabName = "dashboard",
          frow
    ),
    tabItem(tabName = "model",
            "Model Health Dashboard -- COMING SOON"
    )
  #)
)

#completing the ui part with dashboardPage
ui <- dashboardPage(header, sidebar, body, skin='red')


##################################################################################
###################################### SERVER ####################################
##################################################################################


# create the server functions for the dashboard  
server <- function(input, output) { 
  
  # output$churn <- renderGauge({
  #   
  #   req(input$custID)
  #   
    # selected_customer_id <- test_tbl_with_ids$customerID[1]
    # selected_customer_id <- input$customer_id
    # 
    # test_tbl_with_ids_predictions <- test_tbl_with_ids %>% 
    #   mutate(churn_prob = predictions$Yes)
    # 
    # customer_tbl <- test_tbl_with_ids_predictions %>% 
    #   filter(customerID == selected_customer_id)
    # 
    
    
    # df$rank <- order(df$prob)
    # rmax <- max(df$rank)
    # currank <- filter(df, custid == input$custID)$rank
    # gauge <- data.frame(matrix(nrow=1, ncol = 1))
    # names(gauge) <- c("rank")
    # gauge$rank <- round(currank/rmax,3)
    # gauge <- gauge %>% mutate(group=ifelse(rank <0.3, "green",
    #                                  ifelse(rank>=0.3 & rank<0.7, "orange","red")),
    #                     label=paste0(rank*100, "%"),
    #                     title=paste("Customer ", as.character(input$custID), sep=""))
    
  #   gauge(
  #     gauge$rank, 
  #     min = 0, 
  #     max = 100,
  #     gaugeSectors(
  #       success = c(0,33),
  #       warning = c(33, 66),
  #       danger = c(67,100)
  #     ),
  #     symbol = "%"
  #   )
  # })
  
  
  output$churn <- renderPlot({

    df$rank <- order(df$prob)
    rmax <- max(df$rank)
    currank <- filter(df, custid == input$custID)$rank
    gauge <- data.frame(matrix(nrow=1, ncol = 1))
    names(gauge) <- c("rank")
    gauge$rank <- round(currank/rmax,2)
    gauge <- gauge %>% mutate(group=ifelse(rank <0.33, "green",
                                     ifelse(rank>=0.33 & rank<0.67, "orange","red")),
                        label=paste0(rank*100, "%"),
                        title=paste("Customer ", as.character(input$custID), sep=""))

    ggplot(gauge, aes(fill = group, ymax = rank, ymin = 0, xmax = 2, xmin = 1)) +
      geom_rect(aes(ymax=1, ymin=0, xmax=2, xmin=1), fill ="#ece8bd") +
      geom_rect() +
      coord_polar(theta = "y",start=-pi/2) + xlim(c(0, 2)) + ylim(c(0,2)) +
      geom_text(aes(x = 0, y = 0, label = label, colour=group), size=6.5, family="Poppins SemiBold") +
      geom_text(aes(x=1.5, y=1.5, label=title), family="Poppins Light", size=5.2) +
      #facet_wrap(~title, ncol = 1) +
      theme_void() +
      scale_fill_manual(values = c("red"="#C9146C", "orange"="#DA9112", "green"="#129188")) +
      scale_colour_manual(values = c("red"="#C9146C", "orange"="#DA9112", "green"="#129188")) +
      theme(strip.background = element_blank(),
            strip.text.x = element_blank()) +
      guides(fill=FALSE) +
      guides(colour=FALSE) #+ 
      #theme(plot.background = element_rect(fill = "blue"))
  })

  #some data manipulation to derive the values of KPI boxes
  #lastbill <- filter(df, custid == input$custID)$bill
  #age <- filter(df, custid == input$custID)$age
  #datausage <- filter(df, custid == input$custID)$datause
  #creating the valueBoxOutput content
  
  
  output$value1 <- renderValueBox({
    valueBox(
      formatC(filter(df, custid == input$custID)$age, format="d", big.mark=',')
      ,'Age'
      ,icon = icon("stats",lib='glyphicon')
      ,color = "purple")  
  })
  
  
  output$value2 <- renderValueBox({ 
    valueBox(
      formatC(filter(df, custid == input$custID)$bill, format="d", big.mark=',')
      ,'Last Bill'
      ,icon = icon("usd",lib='glyphicon')
      ,color = "green")  
  })
  # output$value3 <- renderValueBox({ 
  #   valueBox(
  #     formatC(filter(df, custid == input$custID)$datause, format="d", big.mark=',')
  #     ,'Data Used Last Month'
  #     ,icon = icon("cog",lib='glyphicon')
  #     ,color = "red")  
  # })
  
  
  #dropperc
  output$dropperc <- renderPlot({
    x <- df$dropperc
    bw <-  (2 * IQR(x)) / length(x)^(1/3)
    value <- round(filter(df, custid == input$custID)$dropperc,5)
    nb <- ceiling(diff(range(x))/bw)
    hl <- round((value - (min(x) - bw)) /bw)
    colors <- c(rep("grey45",hl-1), rep("red",1), rep("grey45",nb-hl))
    x_title <- paste("% Dropped = ", as.character(value), sep="")
    ggplot() + geom_histogram(aes(x), fill=colors, bins = nb) +
      theme(axis.line=element_blank(),
            axis.text.y=element_blank(),axis.ticks.y=element_blank(),
            axis.title.y=element_blank(),legend.position="none")+
      labs(x = x_title)
  })
  
  
  #mins
  output$mins <- renderPlot({
    x <- df$mins
    bw <-  (2 * IQR(x)) / length(x)^(1/3)
    value <- filter(df, custid == input$custID)$mins
    nb <- ceiling(diff(range(x))/bw)
    hl <- round((value - (min(x) - bw)) /bw)
    colors <- c(rep("grey45",hl-1), rep("red",1), rep("grey45",nb-hl))
    x_title <- paste("Mins = ", as.character(value), sep="")
    ggplot() + geom_histogram(aes(x), fill=colors, bins = nb) +
      theme(axis.line=element_blank(),
            axis.text.y=element_blank(),axis.ticks.y=element_blank(),
            axis.title.y=element_blank(),legend.position="none")+
      labs(x = x_title)
  })
  
  
  #consecmonths
  output$consecmonths <- renderPlot({
    x <- df$consecmonths
    bw <-  (2 * IQR(x)) / length(x)^(1/3)
    value <- filter(df, custid == input$custID)$consecmonths
    nb <- ceiling(diff(range(x))/bw)
    hl <- round((value - (min(x) - bw)) /bw)
    colors <- c(rep("grey45",hl-1), rep("red",1), rep("grey45",nb-hl))
    x_title <- paste("Months = ", as.character(value), sep="")
    ggplot() + geom_histogram(aes(x), fill=colors, bins = nb) +
      theme(axis.line=element_blank(),
            axis.text.y=element_blank(),axis.ticks.y=element_blank(),
            axis.title.y=element_blank(),legend.position="none")+
      labs(x = x_title)
  })
  
  
  #income
  output$income <- renderPlot({
    x <- df$income
    bw <-  (2 * IQR(x)) / length(x)^(1/3)
    value <- filter(df, custid == input$custID)$income
    nb <- ceiling(diff(range(x))/bw)
    hl <- round((value - (min(x) - bw)) /bw)
    colors <- c(rep("grey45",hl-1), rep("red",1), rep("grey45",nb-hl))
    x_title <- paste("Income = $", as.character(value), "K", sep="")
    ggplot() + geom_histogram(aes(x), fill=colors, bins = nb) +
      theme(axis.line=element_blank(),
            axis.text.y=element_blank(),axis.ticks.y=element_blank(),
            axis.title.y=element_blank(),legend.position="none")+
      labs(x = x_title)
  })
}



#run/call the shiny app
shinyApp(ui, server)