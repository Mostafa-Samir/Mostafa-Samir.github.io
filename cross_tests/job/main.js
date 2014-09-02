var app = {};
app.control = {
  checked_radio_btn : null,
};

app.renderChecked = function(btn) {
  var real;
  if(utilis.badBrowser())
    real = utilis.nextElementSibling(utilis.nextElementSibling(btn));
  else
    real = utilis.nextElementSibling(btn);
  real.style.border = "1px solid #5F2BC1";
  real.style.backgroundColor = "#F1F1F1";
  real.insertAdjacentHTML("beforeend", "<span class='inner-circle'></span>");

  if(btn.value === "other")
    document.querySelector('.no-feedback textarea').disabled = false;
}

app.renderUnChecked = function(btn) {
  var real;
  if(utilis.badBrowser())
    real = utilis.nextElementSibling(utilis.nextElementSibling(btn));
  else
    real = utilis.nextElementSibling(btn);
  real.style.border = "1px solid gray";
  real.style.backgroundColor = "white";
  real.removeChild(document.querySelector('.inner-circle'));

  if(btn.value === "other")
    document.querySelector('.no-feedback textarea').disabled = true;
}

app.onHelpful = function() {
  document.querySelector("#qhelpful").style.display = "none";
  document.querySelector(".thanks").style.display = "block";
}

app.onNotHelpful = function() {
  document.querySelector("#qhelpful").style.display = "none";
  document.querySelector(".no-feedback").style.display = "block";
}

app.macthColumns = function() {
  var col_60 = document.querySelector('.content');
  var col_40 = (function(){var all = document.querySelectorAll('.menu'); return all[all.length - 1];})();

  //reset to intial before resizing to avoid unneccesary extensions
  col_60.style.height = "auto";
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
  shorter.style.height = shorter.offsetHeight + dh + "px";
}

app.init = function() {
  //creating the customized radio buttons
  utilis.forEach(document.querySelectorAll('.radio-btn'), function (btn) {
    btn.insertAdjacentHTML("afterend", "<span class='cust-radio-btn'></span>");

    if(btn.checked) {
      app.renderChecked(btn);
      app.control.checked_radio_btn = btn;
    }

    btn.onchange = function() {
      app.renderUnChecked(app.control.checked_radio_btn);
      app.renderChecked(btn);
      app.control.checked_radio_btn = btn;
    };

    if(utilis.badBrowser())
      btn.onclick = function() {
        btn.blur();
        btn.focus();
      }

  });

  //mimiking placeholder for the explaination textarea in ie8 and ie9
  var txtarea = document.querySelector("#other_explain");
  if(typeof txtarea.placeholder === "undefined")
    utilis.setPlaceholder(txtarea, "Please Explain...");

  //binding buttons events
  document.querySelector("#nohelpful").onclick = app.onNotHelpful;
  document.querySelector("#helpful").onclick = app.onHelpful;

  app.macthColumns();
};

window.onload = app.init;
window.addEventListener('resize' , app.macthColumns);
