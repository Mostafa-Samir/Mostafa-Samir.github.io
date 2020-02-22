/**
 ** @author: Mostafa Samir
 ** @email: mostafa.3210@gmail.com
 ** @liscence: MIT
 **
**/

'use strict';

/**
 * creates a minimal video controler on video elements
 * @param {HTMLElement} block: the block to make expandable
 * block is expected to be a div containing an <h*> element at the beginning
 */
function makeExpandable(block) {
    let firstChild = block.firstChild();
    let firstChildType = first_child.nodeName;

    let headingRegex = /H\d/;

    if (headingRegex.test(firstChildType)) {}
    else {
        throw Error("First element of an expandable block must be a heading");
    }
}