---
layout: post
categories: [asynchronous programming, nodejs, javascript]
title: Asynchronous Iterative & Recursive Patterns for Node.js - Part 1
---

## Introduction

If any thing made node.js as popular as it is now, it'd would its non-blocking asynchronous I/O calls. You baiscally call an I/O routine, give it a function to call back when your operation is done and you go on continuing whatever business you still have without waiting (blocking) for that operation to finish. How simple is that ?!

Well, it turns out that it's not that simple when iterations come in picture. In a lot of situations we may find ourselves in need to iterate over many I/O calls before we can do some specific task (or invoke some callback function), like when we need to read multiple files off disk each of which has some data that contribute to some final info we must present to the user.

Such problems may be - at first - challenging to implement an asynchronous solution for them, but by the end of this two parts article I hope that it'll become clear how they could be solved. We'll investigate two problems of the kind; an iterative one and a recursive one, and as a solution we'll provide some patterns that could be used against any problems of the same type. In this part we'll start with the iterative problem.

## Problem #1
This is a variation of the one we mentioned in the introduction.

> You're given a list of paths to files on disk. Each file conatins a number and you need to output the final summation of all these numbers.

### A Synchronous Solution

```javascript
var fs = require('fs');

var paths = [
             '/path/to/first-file',  // contains 10
             '/path/to/second-file',  // contains 7
             '/path/to/third-file'  // conatins 5
             ];
var totalSum = 0;

// this the callback we need to call after all iteartion finish
function PrintTotalSum() { console.log(totalSum); }

for(var i = 0; i < paths.length; i++) {
    var num = parseInt(fs.readFileSync(paths[i], 'utf8'));
    totalSum += num;
}

// invoke the callback
PrintTotalSum();
```

Well, this solves it ! But that's a synchronous (aka, blocking) solution, that's bad ! Imagine that this runs within a server application as a respond to a request, while the server is processing one request all the other requests are blocked, the server will only be able to handle one client at a time. Not cool !

We need an asynchronous solution. Here's a wrong one !

### A Wrong Asynchronous Solution

```javascript
for(var i = 0; i < paths.length; i++) {

    fs.readFile(paths[i], 'utf8', function(err, data) {
        var num = parseInt(data);
        totalSum += num;
    });
}

// invoke the callback
PrintTotalSum();
```

I encourge you to whip up three files with contents like the example here and test that yourself, you're most likely gonna get a 0.

So why is this wrong ? We used the asynchronous `fs.readFile` so what's the problem ?!

The Problem is that we used the asynchronous `fs.readFile` with the for-loop. At each iteration of the for-loop, a file is requested to be read  from disk, and because this is an asynchronous request the for-loop doesn't wait for the respond and goes on to the next iteration and request the next file. As soon as the for-loop ends, the line calling the `PrintTotalSum` function will be executed immediatly while the files are still being read. At the time `PrintTotalSum` is executed, it finds the value of `totalSum` to be 0, and so a 0 is what you get.

### A Correct Asynchronous Solution

Let's imagine that you're gonna simulate this problem with three of your friends. You give each of them one of the files and ask him/her to read the number off the file then put in a bag at your disk a number of balls equal to the number he/she read off the file. You then leave, but before that you give each of them a phone number to call you on and **report** that he/she finsihed the job, so that when recive three calls you go back and deliver the bag full of balls to the customer who requested it.

This is essentially the correct asynchronous solution to the problem. We could implement this in a pattern that we can use with any problem of the same class.

```javascript
function IterateOver(list, iterator, callback) {
    // this is the function that will start all the jobs
    // list is the collections of item we want to iterate over
    // iterator is a function representing the job when want done on each item
    // callback is the function we want to call when all iterations are over

    var doneCount = 0;  // here we'll keep track of how many reports we've got

    function report() {
        // this function resembles the phone number in the analogy above
        // given to each call of the iterator so it can report its completion

        doneCount++;

        // if doneCount equals the number of items in list, then we're done
        if(doneCount === list.length)
            callback();
    }

    // here we give each iteration its job
    for(var i = 0; i < list.length; i++) {
        // iterator takes 2 arguments, an item to work on and report function
        iterator(list[i], report)
    }
}
```

Using this, we can simply solve our problem by calling `IterateOver` on our array of paths and define the iterator to be the body of the for-loop we wrote in the wrong solution with the addition of the call to `report` after adding the number to the total to report that the iteartion is over. And we pass the `PrintTotalSum` as the callback. This will asynchronously always get you a 22.

```javascript
IterateOver(paths, function(path, report) {
    fs.readFile(path, 'utf8', function(err, data) {

        var num = parseInt(data);
        totalSum += num;

        // we must call report to report back iteration completion
        report();
    });
}, PrintTotalSum);
```

## Problem #1.2

Problem #1 didn't impose any order on the processing of the list of paths, it doesn't matter which file will be read first because the addition's result will always be the same. Lets re-formulate it a little so that the order of processing matters and see what happens.

> You're given a list of paths to files on disk. Each file conatins a number and you need to output one string which contains all the numbers read form the files in the order of the paths in the list. However, on each read you must wait a random time that ranges from 0-10 milliseconds before appending the read number to the string.

### A Wrong Solution

We could try the `IterateOver` pattern we created on this problem

```javascript
var FinalResult = "";

IterateOver(paths, function(path, report) {
    fs.readFile(path, 'utf8', function(err, data) {

        // here we wait for random time
        setTimeout(function() {
            FinalResult += data + " ";

            report();
        }, Math.floor(Math.random() * 10));

    });
}, function() {
    console.log(FinalResult);
});
```

If you run this solution you'll sometimes get **"10 7 5"**, and some other times **"7 5 10"** or **"5 10 7"**. The order of number is undetermined and doesn't always match with the order of paths in the list. So What's the problem ?

Again, the problem is with the for-loop in the `IterateOver` function. Each iteration starts an asynchrnous call and doesn't wait for it to finish. So when the for-loop ends you'll have your three asynchrnous calls working together, and becuase of the randomness in the waiting they will probably not finish in order. So what happens is that the one that finishes first appends its number with no regard to the order of calling.

We conclude from this that the `IterateOver` pattern doesn't guarantee order. We want a solution that does.

### A Correct Solution

Let's get back to the simualtion with the three friends. This time you ask each of them to read the number of his/her file and write it down on a piece of paper on your disk, but now you'll stack the files on your disk and want them to write the numbers in the same order of the stack. But - as last time - you must leave so you won't be here to enforce the order.

An simple solution to this is to delegate the order enforcement to your friends. So before you leave, you tell the first one to pick the first file, read it and write it down, then **report** to the next one that he/she should process the next file. You finally instruct them that the last one should **report** back to you that he/she finishes to indicate that all the work is done so that you could return and deliver the paper with all the numbers on it to the customer who asked for it.

We can see that the work will proceed in a waterfall manner where each stage will start the next one until the job is done. We could simply implement this waterfall pattern by tweaking a few parts of the `IterateOver` pattern.

```javascript
function WaterfallOver(list, iterator, callback) {

    var nextItemIndex = 0;  //keep track of the index of the next item to be processed

    function report() {

        nextItemIndex++;

        // if nextItemIndex equals the number of items in list, then we're done
        if(nextItemIndex === list.length)
            callback();
        else
            // otherwise, call the iterator on the next item
            iterator(list[nextItemIndex], report);
    }

    // instead of starting all the iterations, we only start the 1st one
    iterator(list[0], report);
}
```

Now we can use the `WaterfallOver` pattern instead of the `IterateOver` pattern and always get **"10 7 5"** as desired.

```javascript
var FinalResult = "";

WaterfallOver(paths, function(path, report) {
    fs.readFile(path, 'utf8', function(err, data) {

        // here we wait for random time
        setTimeout(function() {
            FinalResult += data + " ";

            report();
        }, Math.floor(Math.random() * 10));

    });
}, function() {
    console.log(FinalResult);
});
```

Before we finish here, there might be a bothering quation about Problem #1.2, particulary about the condition of random waitings. I mean, what kind of condition is that ?! It seems completely arbitrary and cannot be found in rael-life problems !

Well, it does indeed seem arbitrary but it was importent to put here to demonstarte the point of uncertain order of completion. In most real-life problems, your asynchronous calls will actually take differnet amount of time to complete and the order of completion will be really undetermined. But in the example of the three files we were working on here, bacause the files are small and probably close to each other on disk, the order of completion will very likely be the same order of the list and the point I wanted to make wouldn't be appearent. So I put the randomness condition to overcome this. However, when you work on real-life problems you'll likely find your asynchronous calls to be undetermined in their completion order without adding anything.

## Summery

In this part we explored the problem of iterations with the asynchronous programming model in node.js. We introduced two patterns to solve this problem; the `IterateOver` pattern which doesn't guarantee order of processing, and the `WaterfallOver` pattern that does. Next time, we'll explore the problem of recursion with asynchronous jobs and how it could be solved using promises.
