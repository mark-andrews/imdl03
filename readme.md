# Introduction to Machine Learning and Deep Learning using R

In this two day course, we provide an introduction to machine learning
and deep learning using R. We begin by providing an overview of the machine
learning and deep learning landscape, and then turn to some major machine learning applications.
We begin with binary and multiclass classification problems, then
look at decision trees and random forests, then look at unsupervised learning methods, all of which are major topics in
machine learning. We then cover artificial neural networks and deep learning.
For this, we will use the powerful `TensorFlow` and `Keras` deep learning toolboxes. As examples of deep learning nets, we will cover the relatively easy to understand _multilayer perceptron_ and then turn to
_convolutional neural networks_.

## Intended Audience

This course is aimed at anyone who is interested in learning the machine
learning or deep learning using R, both of which have major applications
both in industry and in academia.

## Teaching Format

This course will be hands-on and workshop based. Throughout each day, there will
be some brief introductory remarks for each new topic, introducing and
explaining key concepts.

The course will take place online using Zoom. On each day, the live video
broadcasts will occur between (UK local time) at:

-   10am-12pm
-   1pm-3pm
-   5pm-6pm

All sessions will be video recorded and made available to all attendees as soon
as possible, hopefully soon after each 2hr session.

Attendees in different time zones will be able to join in to some of these live
broadcasts, even if all of them are not convenient times.

By joining any live sessions that are possible, this will allow attendees to
benefit from asking questions and having discussions, rather than just watching
prerecorded sessions.

Although not strictly required, using a large monitor or preferably even a
second monitor will make the learning experience better.

All the sessions will be video recorded, and made available immediately on a
private video hosting website. Any materials, such as slides, data sets, etc.,
will be shared via GitHub.

## Assumed quantitative knowledge

We will assume familiarity with some general statistical and mathematical
concepts such as matrix algebra, calculus, probability distributions. However,
expertise with these concepts are not necessary. Anyone who has taken any
undergraduate (Bachelor's) level course in mathematics, or even advanced high
school level, can be assumed to have sufficient familiarity with these concepts.

## Assumed computer background

We assume general familiarity with using R and RStudio, and some familiarity with programming in R, such as writing code.

## Equipment and software requirements

Attendees of the course must use a computer with R/RStudio installed, as well as the necessary additional R packages.
Instructions on how to install and configure all the software are [here](software.md).

# Course programme

## Day 1

-   Topic 1: _Machine learning and Deep Learning Landscape_. Concepts like machine
    learning, deep learning, big data, data science have become widely used and
    celebrated in the last 10 years. However, their definitions are relatively
    nebulous, and how they related to one another and to major fields like
    artificial intelligence and general statistics are not simple matters. In this
    introduction, we briefly overview the field of machine learning and deep
    learning, discussing its major characteristics and sub-topics.

-   Topic 2: _Classification problems_. Classification problems is one of the
    bread and butter topics in machine learning, and is usually the first topic
    covered in introductions to machine learning. Here, we will cover image
    classification (itself a "hello world" type problem in machine learning), and
    particularly focus on logistic regression and support vector machines (SVMs).

-   Topic 3: _Decision trees and random forests_. Decision trees are a widely  
    used machine learning method, particularly for classification. Random forests  
    are an _ensemble learning_ extension of decision trees whereby large number of
    decision tree classifiers are aggregated, which often leads to much improved
    performance.

## Day 2

-   Topic 4: _Unsupervised machine learning_. Unsupervised learning is essentially
    a method of finding patterns in unclassified data. Here, we will look at two of
    the most widely used unsupervised techniques: k-means clustering and
    probabilistic mixture models.

-   Topic 5: _Introducing artificial neural networks and deep learning_. R provides many packages for artificial neural networks and deep learning. These include Keras and TensorFlow, which are in fact interfaces to Python packages. These are the most widely used major packages for deep learning in R. More recently, native support for deep learning using R via `Torch` has been introduced. We will discuss this, but our focus will be on Keras and TensorFlow given that widespread use.

-   Topic 6: _Multilayer perceptons_. Multilayer perceptrons are very powerful,
    yet relatively easy to understand, artificial neural networks. They are also the
    simplest type of deep learning model. Here, we will build and train a
    multilayer perceptron for a classification problem.

-   Topic 7: _Convolutional neural networks_. Convolutional neural networks (CNNs)
    have proved high successful at image classification, primarily due to their
    _translation invariance_, which is highly suitable for computational vision.
    Here, we use PyTorch to build and train a CNN for image classification.
