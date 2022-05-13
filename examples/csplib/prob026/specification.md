---
Title:    Sports Tournament Scheduling
Proposer: Toby Walsh
Category: Scheduling and related problems
---

The problem is to schedule a tournament of $n$ teams over $n-1$ weeks, with each week divided into $n/2$ periods, and each period divided into two slots. The first team in each slot plays at home, whilst the second plays the first team away. A tournament must satisfy the following three constraints: every team plays once a week; every team plays at most twice in the same period over the tournament; every team plays every other team.

An example schedule for 8 teams is: 

<table>
  <tr>
    <td></td><td>Week 1</td><td>Week 2</td><td>Week 3</td><td>Week 4</td><td>Week 5</td><td>Week 6</td><td>Week 7</td>
  </tr>
  <tr>
    <td>Period 1</td><td>0 v 1</td><td>0 v 2</td><td>4 v 7</td><td>3 v 6</td><td>3 v 7</td><td>1 v 5</td><td>2 v 4</td>
  </tr>
  <tr>
    <td>Period 2</td><td>2 v 3</td><td>1 v 7</td><td>0 v 3</td><td>5 v 7</td><td>1 v 4</td><td>0 v 6</td><td>5 v 6</td>
  </tr>
  <tr>
    <td>Period 3</td><td>4 v 5</td><td>3 v 5</td><td>1 v 6</td><td>0 v 4</td><td>2 v 6</td><td>2 v 7</td><td>0 v 7</td>
  </tr>
  <tr>
    <td>Period 4</td><td>6 v 7</td><td>4 v 6</td><td>2 v 5</td><td>1 v 2</td><td>0 v 5</td><td>3 v 4</td><td>1 v 3</td>
  </tr>
</table>

One extension of the problem is to double round robin tournaments in which each team plays every other team (as before) but now both at home and away. This is often solved by repeating the round robin pattern, but swapping home games for away games in the repeat.