
function getBase64Image(img) {

  var canvas=document.getElementById("c");
    canvas.style.display="none";
    canvas.style.zIndex = "-100";
  canvas.width =img.width;

  canvas.height = img.height;
  console.log(img.width);
  var ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0);
  var dataURL = canvas.toDataURL("image/png");
  canvas.width="0px";
  canvas.height="0px";
  return dataURL.replace(/^data:image\/(png|jpg);base64,/, "");
}


function predictData(){
  var go=true
  var input=document.getElementById("imagetopred");

  var e = document.getElementById("inputGroupSelect01");
var base64 = getBase64Image(document.getElementById("imageid"));
console.log(base64);
var value = e.options[e.selectedIndex].value;
print(value)
var text = e.options[e.selectedIndex].text;
if (text=="Choose Neural Network Model"){
  alert("Please Select a Network")
  go=false
}
if(go){
  var content=document.getElementById("todisplay");

  var load=document.getElementById("duringloading");
  
  content.style.visibility="hidden";
  load.style.visibility="visible";


  var filesToUpload = input.files;

  var xhr = new XMLHttpRequest();

var ctx = c.getContext("2d");
console.log(ctx);
var xhr = new XMLHttpRequest();
if(text=="5 classes VGG16"){
  xhr.open("POST", "http://192.168.56.1:8080/1/");
}
else{
  xhr.open("POST", "http://192.168.56.1:8080/0/");
}

xhr.onload = function () {
  content.style.visibility="visible";;
  load.style.visibility="hidden";    // do something to response
    console.log(xhr.response)
  var result=JSON.parse(xhr.response);
  console.log(result)
    var x = document.getElementById("uploadform");
    x.style.display = "none";
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
