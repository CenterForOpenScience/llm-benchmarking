##=====================================================
## August 19, 2020
##
## This code scrapes number of complaints received by plumbing firms located in Illinois, USA.
##
## List of variables collected:
##      1. web link to each company (bbb.com)
##      2. Company name
##      3. Company address
##      4. Number of complaints (5 categories)
##
##
##
## Part I. Checking and getting BBB.com link to plumbing firms in Illinois
## Using the phone number of registered plumbing firms, attained from ReferenceUSA (Plumbings_All.csv),
## (1) it creates search queries to bbb.com, 
## (2) it collects the list of firms (search results) and link to each firm on bbb.com, and
## (3) it keeps initial search results in InitialScraping_Full.csv 
## 
##
## Part II. Scraping 'complaints' information for each company 
## Using the link to each company collected from Part I, 
## it scrapes the following information for each company (Data_w_RefUSA.csv).
## (1) Company address,
## (2) Company Phone number, and
## (3) Number of complaints received during the most recent three years.
##
##
## Part III: Merging Scraping Results with ReferenceUSA data and removing duplicates (Final_Data_for_Replication.csv)
## (1) Combine scraping results with ReferenceUSA data (company name registered in ReferenceUSA)
## (2) remove duplicated search results
## (3) the resulting dataset will contain 
##     (i) Phone-number (registered in ReferenceUSA)
##     (ii) total complaints registered in BBB.org
##     (iii) dummy variable indicating whether company name begins with "A", number or a symbol (e.g.#)
##
##
## This codes will create the following three datasets:
##     InitialScraping_Full.csv
##     Data_w_RefUSA.csv
##     Final_Data_for_Replication.csv
##=====================================================
rm(list=ls())
library('stringr') # need to manipulate string
library('rvest') # need for data scraping
library('httr')  # need for data scraping
library('sqldf') # need for data screening
setwd("***********")


# Loading data attained from ReferenceUSA
RefUSA <- as.data.frame(read.csv("Plumbings_All.csv"))

# get phone number
PhoneN <- RefUSA$Phone.Number.Combined        # 5628 entries
PhoneN <- PhoneN[PhoneN!="Not Available"]     # 5531 firms with valid phone numbers
PhoneN <- as.character(unique(PhoneN))        # 5217 phone numbers (after removing duplicates)

##########################################################
# Part I: Scraping links to plumbing firms in Illinois
##########################################################
InitialScreen <- rbind.data.frame(c())
for(m in 1:length(PhoneN)){
  Sys.sleep(30) # Giving 30 seconds of break between each query submission
  
  # create search url based on phone number
  PhoneNumber = PhoneN[m]
  url <- paste("https://www.bbb.org/search?find_country=USA&find_loc=il&find_text=",
               str_replace(str_replace(str_replace(PhoneNumber, "\\(","%28"), "\\)","%29"), " ","%20"),
               "&page=1", sep="")
  
  # Read search results
  webpage <- read_html(url)
  
  # scraping company name
  CompanyNameInfo <- html_nodes(webpage,'.gNkQmF')  
  CompanyName<- html_text(CompanyNameInfo)
  
  # scraping web link to each company from search page
  CompanywebLink<- html_attr(CompanyNameInfo,"href")
  
  # Error handling - if there is not search results, record "NA"
  if(length(CompanyName)==0){
    CompanyName <- NA
    CompanywebLink <- NA
    cat(m, " -  Non-Match \n")
  }else{
    cat(m, " -  ", length(CompanywebLink), "\n")
  }
  tmpDT <- cbind.data.frame(N=m, CompnayName=CompanyName, PhoneNumber=PhoneNumber, CompanyLink=CompanywebLink)
  InitialScreen <- rbind.data.frame(InitialScreen,tmpDT)
}
write.csv(InitialScreen,"InitialScraping_Full.csv", row.names=FALSE)




##########################################################
# Part II: Scraping complaints data
##########################################################
InitialScreen <- unique(InitialScreen[!is.na(InitialScreen$CompanyLink),])
link <- paste(as.character(InitialScreen$CompanyLink),"/complaints",sep="")

# Collecting Complaints Begins Here
myscrapedata <- rbind.data.frame(c())
for(i in 1:length(link)){
  print(i)
  Sys.sleep(30) # Giving 30 seconds of break between each query submission
  InitialInfo <- InitialScreen[i,]
  InitialName <- as.character(InitialInfo$CompnayName)
  InitialPhone <- as.character(InitialInfo$PhoneNumber)
  
  # Reading webpage (if it does not exist, return "error")
  webpage <- tryCatch(read_html(link[i]), 
                      error = function(e) return(paste("error"))) 
  
  if(webpage!="error"){
    # Scraping complaints information
    complaintsInfo <- html_nodes(webpage,'.MuiTableCell-alignRight')  
    complaints<- html_text(complaintsInfo)
    if(length(complaints)==0){complaints <- c(NA,NA,NA,NA,NA,NA);}
    # Scraping company address
    addressInfo <- html_nodes(webpage,'.gsXzVS')  
    address<- unique(html_text(addressInfo))
    if(length(address)==0){address <- NA;}
    # Scraping company name (for double check)
    CompanyNameInfo <- html_nodes(webpage,'.dYxHUw')  
    CompanyName <- unique(html_text(CompanyNameInfo))
    if(length(CompanyName)==0){CompanyName <- NA;}
    # combining information (weblink, name, address and 5 categories of complaints)
    
    tmpdt <- cbind.data.frame(N=i, InitialName=InitialName, InitialPhone=InitialPhone,
                              link=link[i], company=CompanyName, address=address,
                              AdvertisingSales=complaints[1],
                              BillingCollections=complaints[2],
                              DeliveryIssues=complaints[3],
                              GuaranteeWarranty=complaints[4],
                              ProductService=complaints[5],
                              Total=complaints[6])
    myscrapedata <- rbind.data.frame(myscrapedata, tmpdt)
  }else{
    print("webpage not found")
  }
}

# Remove Plumbing Firms Located outside of Illinois
myscrapedata <- na.omit(myscrapedata)
myscrapedata <- myscrapedata[str_detect(as.character(myscrapedata$address), ", IL"),]


# Remove Duplicated Entries by URL or Phone Number
myscrapedata <- as.data.frame(sqldf("select myscrapedata.* from myscrapedata group by link"))
myscrapedata <- myscrapedata[,c(2:ncol(myscrapedata))]


##########################################################
# Part III: Merging Scraping Results with ReferenceUSA data and removing duplicates
##########################################################
# Re-Loading data attained from ReferenceUSA
RefUSA <- as.data.frame(read.csv("Plumbings_All.csv"))
RefUSA$PhoneN <- gsub(" ", "",gsub("[[:punct:]]", "", RefUSA$Phone.Number.Combined))


# Adding a column containing Company name registered in ReferenceUSA
# If equal or more than two company names are associated with a phone number, 
# only one company name (first one by alphabetical order) will be recorded.
Data_w_RefUSA <- rbind.data.frame(c())
for(i in 1:nrow(myscrapedata)){
  tmpdt <- myscrapedata[i,]
  number <- gsub(" ", "",gsub("[[:punct:]]", "", tmpdt$InitialPhone))
  RefTmpDT <- subset(RefUSA, RefUSA$PhoneN==number)
  RefName <- as.character(RefTmpDT$Company.Name)
  tmpdt2<-cbind.data.frame(RefUSAComName=RefName, tmpdt)
  Data_w_RefUSA <- rbind.data.frame(Data_w_RefUSA, tmpdt2)
}
write.csv(Data_w_RefUSA, "Data_w_RefUSA.csv", row.names=FALSE)


Data_w_RefUSA_v2 <- Data_w_RefUSA[,c("RefUSAComName","InitialName","InitialPhone","company","address","AdvertisingSales","BillingCollections","DeliveryIssues","GuaranteeWarranty","ProductService","Total")]
Data_w_RefUSA_v2 <- as.data.frame(sqldf("select Data_w_RefUSA_v2.* from Data_w_RefUSA_v2 group by RefUSAComName, InitialPhone"))
Data_w_RefUSA_v2$total2<-0
for(i in 1:nrow(Data_w_RefUSA_v2)){
  tmpdt <- subset(Data_w_RefUSA_v2, Data_w_RefUSA_v2$InitialPhone==as.character(Data_w_RefUSA_v2$InitialPhone[i]))
  Data_w_RefUSA_v2$total2[i] <- sum(tmpdt$Total)
}


Final_dataset <- rbind.data.frame(c())
PhoneNumber <- as.character(unique(Data_w_RefUSA_v2$InitialPhone))
for(i in PhoneNumber){
  tmpdt <- subset(Data_w_RefUSA_v2, Data_w_RefUSA_v2$InitialPhone==i)
  tmpdt$RefUSAComName <- sort(as.character(tmpdt$RefUSAComName))[1]
  Final_dataset <- rbind.data.frame(Final_dataset, tmpdt[1,])
}


Final_dataset$FirstChar <- substr(as.character(gsub(" ", "",Final_dataset$RefUSAComName)),1,1)
Final_dataset$First_A <- 0 + 1*(Final_dataset$FirstChar=="1")+ 1*(Final_dataset$FirstChar=="2")+ 1*(Final_dataset$FirstChar=="3")+ 1*(Final_dataset$FirstChar=="4")+ 1*(Final_dataset$FirstChar=="A")

Final_dataset <- Final_dataset[,c("InitialPhone","total2","First_A")]
colnames(Final_dataset) <- c("firm_id", "complaints","first_A")
write.csv(Final_dataset, "Final_Data_for_Replication.csv", row.names=FALSE)

# End of the Code


