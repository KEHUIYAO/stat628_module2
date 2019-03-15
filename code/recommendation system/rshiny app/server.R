library(shiny)
library(dplyr)
library(ggplot2)
library(leaflet)
library(stringr)

data = read.csv("final.csv", header = TRUE)

server = shinyServer(function(input, output) {
  new_data <- reactive({
    new_data = data[data$business_id == as.numeric(input$B_id),]
    if (is.na(new_data[1,1]) == TRUE){
      new_data = data.frame(business_id = 1)
    }
    new_data
    })
  convertlist_mean = function(list){
    char = gsub('\\[|\\]','',as.character(list))
    x = strsplit(char,', ')[[1]]
    return(round(mean(as.integer(x)),2))
  }
  
  nearby_data <- reactive({
    city = data[data$business_id == as.numeric(input$B_id),]$city
    nearby_data = data[data$city == city,]
    nearby_data$stars = lapply(nearby_data$stars, convertlist_mean)
    nearby_data
  })
  
  convertlist = function(list){
    char = gsub('\\[|\\]','',as.character(list))
    x = strsplit(char,', ')[[1]]
    return(as.integer(x))
  }
  
  output$review_dist = renderPlot({
    if (new_data()$business_id == 1){
      plot.new()
    }else{
      star = convertlist(new_data()$stars)
      df = data.frame(star=star) %>% count(star)
      names(df)[names(df)=='n'] = 'count'
      ggplot(data=df,mapping=aes(x=star,y=count,fill=star,group=factor(1)))+
        geom_bar(stat="identity") + ggtitle("Review Distribution") + 
        theme(plot.title = element_text(hjust = 0.5))
    }
})
  
  convertstring = function(list){
    char = gsub('\\[|\\]','',as.character(list))
    x = strsplit(char,', ')[[1]]
    return(x)
  }
  
  output$text = renderTable({
    review = convertstring(new_data()$reviews)
    df = data.frame(Importance=c("1st","2nd","3rd","4th","5th"), Tag = review)
    df
  })
  
  output$suggestions = renderTable({
    suggestion = as.character(new_data()$comments)
    suggestion = strsplit(suggestion,"\n")[[1]]
    df = data.frame(suggestion = suggestion[1])
    for (i in 2:length(suggestion)){
      if (suggestion[i] != ""){
        df = rbind(df,data.frame(suggestion = suggestion[i]))
      }
    }
    df
  })
  
  output$recommendation = renderTable({
    suggestion = as.character(new_data()$features)
    suggestion = strsplit(suggestion,"\n")[[1]]
    i = 1
    for(x in suggestion){
      if(i == 1){
        title = regmatches(x,regexpr("[A-Z]+", x))
        a = gsub('[A-Z]+ ','',x)
        b = strsplit(a,'; ')[[1]]
        len = length(b)
        tc = c(title,rep(' ',len-1))
        df = data.frame(attribute = tc,words = b)
        i = i+1
      }else{
        title = regmatches(x,regexpr("[A-Z]+", x))
        a = gsub('[A-Z]+ ','',x)
        b = strsplit(a,'; ')[[1]]
        len = length(b)
        tc = c(title,rep(' ',len-1))
        df2 = data.frame(attribute = tc,words = b)
        df = rbind(df,df2)
      }
    }
    df
  })

  output$map = renderLeaflet({
    center_lng = new_data()$longitude
    center_lat = new_data()$latitude
    lng = nearby_data()$longitude
    lat = nearby_data()$latitude
    name = nearby_data()$name
    score = nearby_data()$star
    tag = nearby_data()$reviews
    content <- paste(sep = "<br/>",
                     name,
                     paste("Average Star:", as.character(score)),
                     as.character(tag)
    )
    leaflet(nearby_data()) %>% setView(lng = center_lng, lat = center_lat, zoom = 17) %>% addTiles() %>% addMarkers(lng=lng, lat=lat, popup=content)
  })
  
  output$cheating = renderText({
    if (new_data()$business_id == 1){
      text = ""
    }else{
      diff = new_data()$diff
      if (diff > 1.5){
        text = "The reviews and stars of this restaurant may be inconsistent"
      }else{
        text = " "
      }
    }
    text
  })
})
