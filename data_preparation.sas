ods select none;
proc surveyselect data=sashelp.iris  out=iris2  
                  samprate=0.5  method=srs  outall;
run;
ods select all;
 
ods html file = '/mnt/output.html';
%let k=15;
proc discrim data=iris2(where=(selected=1))   
             test=iris2(where=(selected=0))
             testout=iris2testout
             method=NPAR k=&k 
             listerr crosslisterr; 
      class Species; 
      var SepalLength SepalWidth PetalLength PetalWidth; 
      title2 'Using KNN on Iris Data'; 
run; 
quit;
ods html close;