document.addEventListener('DOMContentLoaded', function() {
    console.log("SHOUT!!!!");
    var dateobj = new Date();
    var currentYear = dateobj.getFullYear();

    document.querySelector("#ccyear").innerHTML = currentYear;
});
