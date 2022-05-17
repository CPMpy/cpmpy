---
Title:    Vessel Loading
Proposer: Ken Brown
Category: Design and configuration
---



Supply vessels transport containers from site to site. The deck area is rectangular. Containers are cuboid, and are laid out in a single layer. All containers are positioned parallel to the sides of the deck. The contents of the containers determine their class. Certain classes of containers are constrained to be separated by minimum distances either along the deck or across the deck. `

The vessel loading decision problem is to determine whether a given set of containers can be positioned on a given deck, without overlapping, and without violating any of the separation constraints. The problem can be modelled as packing of a set of rectangles into a larger rectangle, subject to constraints.

In practice, the layout may be further constrained by the physical loading sequence. Containers are manoeuvred into position from the south east corner. Each successive container in the loading sequence must be positioned so that it touches part of another container or a deck wall both to the north and to the west.