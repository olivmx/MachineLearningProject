setwd("C:\\TestR\\JHS_Machine_Learning")
answers = c("C","A","C","A","A","E","D","D","A","A","B","C","B","A","E","E","A","B","B","B")

pml_write_files = function(answers){
     n = length(answers)
     for(i in 1:n){
          filename = paste("problem_id_",i,".txt", sep="")
          write.table(answers[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
     }
}
pml_write_files(answers)
