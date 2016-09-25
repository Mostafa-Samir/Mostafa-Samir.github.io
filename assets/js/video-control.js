/**
 ** @author: Mostafa Samir
 ** @email: mostafa.3210@gmail.com
 ** @liscence: MIT
 **
 ** @NOTE: the plugin requires the fontawesome library
**/

'use strict';

/**
 * creates a minimal video controler on video elements
 * @param {HTMLElement} vid: the video element
 */
 function minimalVid(vid) {

     vid.loop = true;

     var parent = vid.parentElement;

     var container = document.createElement('div');
     container.style.position = 'relative';
     container.style.cursor = 'pointer';

     var controlDiv = document.createElement('div');
     controlDiv.dataset.state = 'paused';
     controlDiv.style.position = 'absolute';
     controlDiv.style.width = '100%';
     controlDiv.style.height = '100%';
     controlDiv.style.top = 0;
     controlDiv.style.left = 0;
     controlDiv.style.zIndex = 2;
     controlDiv.style.display = 'flex';
     controlDiv.style.justifyContent = 'center';

     var controlBtn = document.createElement('i')
     controlBtn.className = "fa fa-play-circle-o";
     controlBtn.style.color = "black";
     controlBtn.style.alignSelf = 'center';
     controlBtn.style.fontSize = '11em';

     controlDiv.appendChild(controlBtn);
     container.appendChild(controlDiv);
     parent.insertBefore(container, vid);

     container.appendChild(vid);

     controlDiv.addEventListener('click', function() {
         var state = controlDiv.dataset.state;
         if(state === 'paused') {
             controlDiv.dataset.state = 'playing';
             controlBtn.style.display = 'none';
             vid.play();
         }
         else {
             controlDiv.dataset.state = 'paused';
             controlBtn.style.display = 'block';
             vid.pause();
         }
     });
 }
