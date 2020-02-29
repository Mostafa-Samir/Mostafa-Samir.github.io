document.addEventListener('DOMContentLoaded', function() {

    function promoView(query, pageUrl, promoIsClosed) {

        if (!pageUrl.startsWith("/how-machine-learning-works") && !promoIsClosed) {
            console.log(query.matches)
            if (query.matches) {
                mobilePromo.style.display = "none";
                bookPromo.style.display = "flex";
    
            }
            else {
                bookPromo.style.display = "none";
                mobilePromo.style.display = "block";
            }
        }
    }

    let promoIsClosed = false;
    let OthersMediaMatcher = window.matchMedia("(min-width: 1440px)");
    let homeMediaMatcher = window.matchMedia("(min-width: 1670px)");
    let mediaMatcher = null;

    let pageUrl = window.location.pathname;
    let bookPromo = document.querySelector(".book-promo");
    let mobilePromo = document.querySelector(".book-promo-mobile");

    if (pageUrl === "/") {
        mediaMatcher = homeMediaMatcher;
    }
    else {
        mediaMatcher = OthersMediaMatcher
    }

    mediaMatcher.addListener((query) => promoView(query, pageUrl, promoIsClosed))

    document.querySelector("#promo-close").addEventListener('click', function() {
        bookPromo.style.display = "none";
        promoIsClosed = true;
    });

    promoView(mediaMatcher, pageUrl, promoIsClosed);
});


