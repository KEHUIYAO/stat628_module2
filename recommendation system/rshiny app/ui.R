library(shiny)
library(leaflet)

ui = shinyUI(pageWithSidebar(
  headerPanel("Yelp Recommendation System"),
  
  sidebarPanel(
    numericInput("B_id", "Business ID:", min = 0, max = 100000000, value = NA),
    helpText("Enter the business ID you want to query"),
    submitButton("Search"),
    
    h3('Any issue?'),
    h4('Contact with our team!'),
    h4('Shuo Qiang: sqiang@wisc.edu'),
    h4('Zihan Zhou: zzhou342@wisc.edu'),
    h4('Zhendong Zhou: zzhou339@wisc.edu'),
    h4('Kehui Yao: kyao24@wisc.edu')
    ),
  
  mainPanel(tabsetPanel(
    tabPanel("Resturant Customer Review", imageOutput("review_dist"), textOutput("cheating")),
    tabPanel("Recommendation for Customer", tableOutput("text"), tableOutput("recommendation")),
    tabPanel("Suggestions for Resturant", tableOutput("suggestions")),
    tabPanel("Resturant Nearby", leafletOutput("map"))
  ))
))
