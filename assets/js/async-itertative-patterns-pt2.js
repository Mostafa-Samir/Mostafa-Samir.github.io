document.querySelector("#async-recursion").addEventListener('click', function() {
    if(this.dataset.state === 'static') {
        this.dataset.state = 'animated';
        this.src = '/assets/images/async-recursion.gif';
    }
    else {
        this.dataset.state = 'static';
        this.src = '/assets/images/async-recursion-static.png';
    }
});