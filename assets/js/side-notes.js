/**
 ** @author: Mostafa Samir
 ** @email: mostafa.3210@gmail.com
 ** @liscence: MIT
 **
 ** @NOTE: the plugin requires the fontawesome library and bootstrap-native 2.0.2
**/

'use strict';

/**
 * uses info from hidden sidenote elements to create popovers for the additional info
 * @param {HTMLElement} note: the hidden sidenote element
 */
function processSideNoteElement(note) {

    let noteIconElement = document.createElement('i');
    noteIconElement.className = 'fa fa-info-circle';

    let noteAElement = document.createElement('a');
    noteAElement.href = '#';
    noteAElement.className = 'sidenote-super';

    noteAElement.appendChild(noteIconElement);
    let insertedNoteAElement = note.parentNode.insertBefore(noteAElement, note);

    let contentP = document.createElement('p');
    note.className = '';
    contentP.appendChild(note);

    let containerDiv = document.createElement('div');
    containerDiv.addEventListener('click', function(e) {e.stopPropagation()});
    containerDiv.className = 'sidenote-container';
    containerDiv.appendChild(contentP);
    let appendedContainerDiv = document.querySelector('article').appendChild(containerDiv)

    insertedNoteAElement.addEventListener('click', function(e) {
        e.preventDefault();
        e.stopPropagation();

        // hide any other opened side note
        Array.prototype.slice.call(
            document.querySelectorAll('.sidenote-container')
        )
        .forEach(function(note){ note.style.display = "none"});

        let pos = {
            x: insertedNoteAElement.offsetLeft,
            y: insertedNoteAElement.offsetTop
        };

        appendedContainerDiv.style.display = 'block';

        let screenWidth = window.innerWidth;
        let noteWidth = appendedContainerDiv.clientWidth;

        appendedContainerDiv.style.top = (pos.y + 20) + "px";
        if (pos.x + noteWidth < screenWidth - 50) {
            appendedContainerDiv.style.left = pos.x + "px";
        }
        else {
            console.log(screenWidth)
            let diff = (pos.x + noteWidth) - (screenWidth - 50);
            console.log(diff)
            appendedContainerDiv.style.left = (pos.x - diff) + "px";
        }
    });

}
