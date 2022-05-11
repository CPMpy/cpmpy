---
Title:    Template Design
Proposer: Barbara Smith 
Category: 
    - Bin packing
    - Partitioning and related problems
---


This problem arises from a colour printing firm which produces a variety of products from thin board, including cartons for human and animal food and magazine inserts. Food products, for example, are often marketed as a basic brand with several variations (typically flavours). Packaging for such variations usually has the same overall design, in particular the same size and shape, but differs in a small proportion of the text displayed and/or in colour. For instance, two variations of a cat food carton may differ only in that on one is printed 'Chicken Flavour' on a blue background whereas the other has 'Rabbit Flavour' printed on a green background. A typical order is for a variety of quantities of several design variations. Because each variation is identical in dimension, we know in advance exactly how many items can be printed on each mother sheet of board, whose dimensions are largely determined by the dimensions of the printing machinery. Each mother sheet is printed from a template, consisting of a thin aluminium sheet on which the design for several of the variations is etched. The problem is to decide, firstly, how many distinct templates to produce, and secondly, which variations, and how many copies of each, to include on each template.
The following example is based on data from an order for cartons for different varieties of dry cat-food.

Variation 	|	 Order Quantity
-------  	|   --------------
Liver	 	|	 250,000
Rabbit	 	|	 255,000
Tuna	 	|	 260,000
Chicken Twin|	 500,000
Pilchard Twin|	 500,000
Chicken		|	 800,000
Pilchard	|  1,100,000
Total	    |  3,665,000


Each design of carton is made from an identically sized and shaped piece of board. Nine cartons can be printed on each mother sheet, and several different designs can be printed at once, on the same mother sheet. (Hence, at least 407,223 sheets of card will be required to satisfy these order quantities.)
Because in this example there are more slots in each template (9) than there are variations (7), it would be possible to fulfil the order using just one template. This creates an enormous amount of waste card, however. We can reduce the amount of waste by using more templates; with three templates, the amount of waste produced is negligible. The problem is therefore to produce template plans which will minimize the amount of waste produced, for 1 template, 2 templates,... and so on.

It is permissible to work in units of say 1000 cartons, so that the order quantities become 250, 255, etc.

A variant is to allow up to 10% under-production of some designs, if this allows the overall over-production to be reduced. This is not a sensible option for the catfood problem, because it leads to under-production of all the designs.

The optimal solutions for the catfood problem are shown below. For each template, the table gives a list of the number of slots allocated to each design, e.g. [1,1,1,1,1,2,2,] means that 1 slot is allocated to each of the first five designs and two each to the last two.

No. of templates  | Layouts	of each template | No. of Pressings	| Total pressings
----------------- | -----------------------  | ---------------- |
1		   		  | [1,1,1,1,1,2,2]          | 550,000	        | 550,000
2		          | [0,0,0,0,0,2,7]	         | 158,000	        |
    	          | [1,1,1,2,2,2,0]	         | 260,000	        | 418,000
3		          | [0,5,3,0,0,1,0]	         | 51,000	        |  
    	          | [0,0,1,0,0,7,1]	         | 107,000	        |
    	          | [1,0,0,2,2,0,4]	         | 250,000	        | 408,000