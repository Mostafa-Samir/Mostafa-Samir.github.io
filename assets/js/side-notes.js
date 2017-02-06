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

        // mark the note marker as the current viewed note
        insertedNoteAElement.className = 'sidenote-super sidenote-viewed';

        // hide any other opened side note if any
        let visibleNote = document.querySelector('.sidenote-visible');
        if(visibleNote)  {
            visibleNote.className = "sidenote-container";
            visibleNote.style.display = "none";
        }

        let pos = {
            x: insertedNoteAElement.offsetLeft,
            y: insertedNoteAElement.offsetTop
        };

        // show the note and bring it to the top-left corner to assume its
        // full possible width
        appendedContainerDiv.style.top = "0px";
        appendedContainerDiv.style.left = "0px";
        appendedContainerDiv.style.display = 'block';
        appendedContainerDiv.className = "sidenote-container sidenote-visible"

        let screenWidth = screen.width;
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

/**
 * repositions a visible side note to a new screen size
 * @param {HTMLElement} visibleNote: the note element to position
 */
function repositionNote(visibleNote) {

    // hide and bring the node to top left corner and show it again
    // to assume its full width in the new screen size
    visibleNote.style.display = "none";
    visibleNote.style.top = "0px";
    visibleNote.style.left = "0px";
    visibleNote.style.display = "block";

    let viewdNoteMark = document.querySelector('.sidenote-viewed');
    let pos = {
        x: viewdNoteMark.offsetLeft,
        y: viewdNoteMark.offsetTop
    }

    let screenWidth = screen.width;
    let noteWidth = visibleNote.clientWidth;

    visibleNote.style.top = (pos.y + 20) + "px";
    if (pos.x + noteWidth < screenWidth - 50) {
        visibleNote.style.left = pos.x + "px";
    }
    else {
        let diff = (pos.x + noteWidth) - (screenWidth - 50);
        visibleNote.style.left = (pos.x - diff) + "px";
    }
}
