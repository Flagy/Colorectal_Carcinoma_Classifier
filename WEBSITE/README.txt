This folder contains all file to launch website.
First of all maintain website folder inside the master Banch of the project.
Folder:
\images --> Contains Images for site mockup
\network--> contains files to load networks
Files:
\aboutus.html --> Html page about the team
\index.html--> Index page of the website
\main.css --> Stylesheet for the site
\photo.jpg--> image for debug
\reconstructed-jpg --> same of photo.jpg but reconstruction after base 64 conversion
\telegram.html--> html page about telegram information
\predict.html-->html to perform online prediction
\server.py --> Serving th WEBSITE folder on your localhost IP 
\prova.js --> Functions for image upload
\script.js --> post image information on the website


HOW TO LAUNCH:
  1. Open script.js and modify ip var in line 1 with the IP Address of the machine which launchs server.py
  2. Launch server.py 
  3. go on "ipaddress:8080/index.html" where ipaddress is the one described in point 1
  4. enjoy the service
  
  IF ERROR about Cross Domain Origin apperas, please install CORS on Chrome and enable it to solve the problem
