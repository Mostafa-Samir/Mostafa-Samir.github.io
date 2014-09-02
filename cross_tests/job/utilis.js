/**
* author : Mostafa Samir (mostafa.3210@gmail.com)
* motivation : providing interfaces and features to work with IE8 quirks
**/

var utilis = {};

/*
* a forEach interface to iterate over query selector results in all borowsers as
* IE8 doesn't support native ES5 forEach
* @param container [Array | NodeList] : the container to iterate over
* @param action [Function] : a function indicating action to be performed on elements
*/
utilis.forEach = function(container, action) {
  var len = container.length;
  for(var itr = 0; itr < len; itr++) {
    action(container[itr]);
  }
};

/*
* detects if the browser is a bad browser (namely, IE8-)
* @retunrs [Boolean] : true if less than IE9, false otherwise
*/
utilis.badBrowser = function() {
  if(navigator.appName.indexOf("Internet Explorer") !== -1)
    return navigator.appVersion.indexOf("MSIE 9") === -1 && navigator.appVersion.indexOf("MSIE 1") === -1;
  return false;
}

/*
* nextElementSibling interface for DOM manipulations in all borowsers
* as IE8 doesn't support native nextElementSibling
* @param element [HTMLElement] : the element to get its next sibling in the dom tree
* @returns [HTMLElement] : the next sibling
*/
utilis.nextElementSibling = function(element) {
  return element.nextElementSibling || element.nextSibling;
}

/*
* a function to load css files dynamically to address browser specific stylesheets
* @param cssfile [String] : the css file to load
*/
utilis.loadStyleSheet = function(cssfile) {
  var link = document.createElement("link");
  link.setAttribute("rel", "stylesheet");
  link.setAttribute("type", "text/css");
  link.setAttribute("href", cssfile);

  document.getElementsByTagName("head")[0].appendChild(link);
}

/*
* a function to add the placeholder behaviour in ie8 and ie9
* @param element [HTMLElement] : the element to add the placeholder behaviour to
* @param msg [String] : the placeholder message
*/
utilis.setPlaceholder = function(element, msg) {
  //initialize placeholder layout and set control
  element.value = msg;
  element.style.color = 'gray';
  element._utilis = {
    on_hold : true
  }

  //bind events to mimik placeholder behaviour on content change
  element.onkeyup = function(e) {
    var keyCode = e.keyCode ? e.keyCode : e.which;
    if(element.value === "" || (keyCode === 37 || keyCode === 39 || keyCode === 46)) {
      element.value = "Please Explain...";
      element.style.color = 'gray';
      element._utilis.on_hold = true;
      element.setSelectionRange(0,0);
    }
    else
      element._utilis.on_hold = false;
  };

  element.onkeypress = function() {
    if(element._utilis.on_hold)
      element.value = "";
    element.style.color = "black";
  };

  element.onclick = function(event) {
    if(element._utilis.on_hold) {
      event.preventDefault();
      element.setSelectionRange(0,0);
    }
  }
}
