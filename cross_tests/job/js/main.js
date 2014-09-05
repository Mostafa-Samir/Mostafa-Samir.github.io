var app = {}; //Namespacing all UI realted functions

/*
* a data structure to hold control data and flags related to UI
*/
app.control = {
  checked_radio_btn : null,
};

/*
* render the custom checked radio button layout
* @param btn [HTMLElement] : the radio button to render the custom layout on
*/
app.renderChecked = function(btn) {
  var real;
  if(utilis.badBrowser())
    real = utilis.nextElementSibling(utilis.nextElementSibling(btn)); //for the existence of css3pie element
  else
    real = utilis.nextElementSibling(btn);
  real.style.border = "1px solid #5F2BC1";
  real.style.backgroundColor = "#F1F1F1";
  real.insertAdjacentHTML("beforeend", "<span class='inner-circle'></span>");

  if(btn.value === "other")
    document.querySelector('.no-feedback textarea').disabled = false;
}

/*
* render the custom un-checked radio button layout
* @param btn [HTMLElement] : the radio button to render the custom layout on
*/
app.renderUnChecked = function(btn) {
  var real;
  if(utilis.badBrowser())
    real = utilis.nextElementSibling(utilis.nextElementSibling(btn)); //for the existence of css3pie element
  else
    real = utilis.nextElementSibling(btn);
  real.style.border = "1px solid gray";
  real.style.backgroundColor = "white";
  real.removeChild(document.querySelector('.inner-circle'));

  if(btn.value === "other")
    document.querySelector('.no-feedback textarea').disabled = true;
}

/*
* the event handler for the "Yes" button in the question about article's helpfulness
*/
app.onHelpful = function() {
  document.querySelector("#qhelpful").style.display = "none";
  document.querySelector(".thanks").style.display = "block";
}

/*
* the event handler for the "No" button in the question about article's helpfulness
*/
app.onNotHelpful = function() {
  document.querySelector("#qhelpful").style.display = "none";
  document.querySelector(".no-feedback").style.display = "block";
}

/*
* match the end of the 'content' wrapper in col-60 with the end of
* the last menu item in cok-40
*/
app.macthColumns = function() {
  var col_60 = document.querySelector('.content');
  var col_40 = (function(){var all = document.querySelectorAll('.menu'); return all[all.length - 1];})();

  //reset to intial before resizing to avoid unneccesary extensions
  col_60.style.minHeight = 0 + "px";
  col_60.style.height = "auto";
  col_40.style.minHeight = 0 + "px";
  col_40.style.height = "auto";

  var taller, shorter;
  if(col_60.offsetTop + col_60.offsetHeight > col_40.offsetTop + col_40.offsetHeight) {
    taller = col_60;
    shorter = col_40;
  }
  else {
    taller = col_40;
    shorter = col_60;
  }

  var dh = (taller.offsetTop + taller.offsetHeight) - (shorter.offsetTop + shorter.offsetHeight);
  shorter.style.minHeight = shorter.offsetHeight + dh + "px";
}

/*
* include video files in the html , if browser is not bad (not IE8-)
* an HTML5 video is included, else a youtube link fallback is used
*/
app.includeVideos = function() {
  utilis.forEach(document.querySelectorAll('.vid'), function(vid) {
    var video_element = document.createElement('video');
    if(video_element.canPlayType && video_element.canPlayType('video/mp4') !== "no") {
      video_element.setAttribute('controls', true);
      video_element.setAttribute('width', '100%');
      vid.appendChild(video_element);
      var source = document.createElement('source');
      source.setAttribute('src', vid.getAttribute('data-src'));
      video_element.appendChild(source);
      /*
      * matching the heights after the video element is playbale as the height
      * change after the video tag has loaded the video and displayed the thumbnail
      */
      video_element.oncanplay = app.macthColumns;
    }
    else {
      var youtube_link = vid.getAttribute('data-youtube-fallback');
      var begin = youtube_link.indexOf('v=');
      var end = youtube_link.indexOf('&') !== -1 ? youtube_link.indexOf('&') : youtube_link.length;
      var video_id = youtube_link.substring(begin + 2, end);
      var embed = document.createElement('iframe');
      embed.setAttribute('src', 'http://www.youtube.com/embed/' + video_id);
      embed.setAttribute('height', 315);
      embed.setAttribute('frameborder', 0);
      vid.appendChild(embed);
    }
  });
}

/*
* intialize the layout
*/
app.init = function() {
  //creating the customized radio buttons
  utilis.forEach(document.querySelectorAll('.radio-btn'), function (btn) {
    btn.insertAdjacentHTML("afterend", "<span class='cust-radio-btn'></span>");

    //rendering checked layout for the default checked button
    if(btn.checked) {
      app.renderChecked(btn);
      app.control.checked_radio_btn = btn;
    }

  //binding the onchange button
    btn.onchange = function() {
      app.renderUnChecked(app.control.checked_radio_btn);
      app.renderChecked(btn);
      app.control.checked_radio_btn = btn;
    };

    if(utilis.badBrowser())
      /*
      *forcing the button to blur and focus on click as bad browsers (IE8-) do not trigger
      *the change event unless the element is focused/blured
      */
      btn.onclick = function() {
        btn.blur();
        btn.focus();
      }

  });

  //include video files
  app.includeVideos();


  //mimiking placeholder for the explaination textarea in ie8 and ie9
  var txtarea = document.querySelector("#other_explain");
  if(typeof txtarea.placeholder === "undefined")
    utilis.setPlaceholder(txtarea, "Please Explain...");

  //binding buttons events
  document.querySelector("#nohelpful").onclick = app.onNotHelpful;
  document.querySelector("#helpful").onclick = app.onHelpful;

  //match the heights of the cloumns after all the page is loaded
  app.macthColumns();
};

//binding app.init to the window load event and app.macthColumns to resize event
if(!utilis.badBrowser()) {
  window.addEventListener('load', app.init);
  window.addEventListener('resize' , app.macthColumns);
}
else {
  //bad browsers doesn't support addEventListener
  window.attachEvent('onload', app.init);
  window.attachEvent('onresize' , app.macthColumns);
}
