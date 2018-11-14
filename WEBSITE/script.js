
function getBase64Image(img) {

  var canvas=document.getElementById("c"); //get canvas in the main page
    canvas.style.display="none";//set style to not display it
    canvas.style.zIndex = "-100";//setting index to -100
    canvas.width =img.width;//adapt canvas width with image with
    canvas.height = img.height;
    var ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0);//draw the image on the canvas
    var dataURL = canvas.toDataURL("image/png");//give data from the canvas
    canvas.width="0px";
    canvas.height="0px";
    return dataURL.replace(/^data:image\/(png|jpg);base64,/, "");
}


function predictData(){
  var go=true;//flag checking fields are correctly compiled
  var input=document.getElementById("imagetopred");
  var e = document.getElementById("inputGroupSelect01");
  var base64 = getBase64Image(document.getElementById("imageid"));//conversion for the image in base 64 format
  var value = e.options[e.selectedIndex].value; //value from the dropdown (0-1-2)
  var text = e.options[e.selectedIndex].text;//text  value for the dropdown
  if (text=="Choose Neural Network Model"){//user has not choice the correct option, he has to choose a correct format for the prediction
      alert("Please Select a Network");//alert triggered
      go=false// var gets false value and blocks the processing
}
  if(go){//correct procedure
    var content=document.getElementById("todisplay");//data content of the page
    var load=document.getElementById("duringloading");//loading icon during processing
    content.style.visibility="hidden";//hiding content of page
    load.style.visibility="visible";//displaying the loading icon
    var filesToUpload = input.files;
    var xhr = new XMLHttpRequest();
    var ctx = c.getContext("2d");
    console.log(ctx);
    var xhr = new XMLHttpRequest();//open HTTP request
    if(text=="5 classes VGG16"){
      xhr.open("POST", "http://192.168.56.1:8080/1/");
}
    else{
      xhr.open("POST", "http://192.168.56.1:8080/0/");
}
//changing url based on user choice
  xhr.onload = function () {//when post procedure is finished...
    content.style.visibility="visible";;//display content
    load.style.visibility="hidden";    // hiding loading icon
    console.log(xhr.response)
    var result=JSON.parse(xhr.response);//parsing the result
    console.log(result)
    var x = document.getElementById("uploadform");
    x.style.display = "none";//hiding the load form
    //creating data showing results
    var y=document.getElementById("result");
    var title=document.createElement("H1");
    title.innerHTML=result.typeofnet;
    y.appendChild(title);
    var h1=document.createElement("H3");
    h1.innerHTML="The network obtained an accuracy of "+result.percentage +" and classified your image in class "+result.class;
    y.appendChild(h1);

};
xhr.send(base64);

}


}
