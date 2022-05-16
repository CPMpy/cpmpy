---
Title:    Word Design for DNA Computing on Surfaces
Proposer: Marc van Dongen
Category: 
    - Bioinformatics
    - Combinatorial mathematics
---


This problem has its roots in Bioinformatics and Coding Theory.

Problem: find as large a set *S* of strings (words) of length *8* over the alphabet *W = { A,C,G,T }* with the following properties:

- Each word in *S* has *4* symbols from *{ C,G }*;
- Each pair of distinct words in *S* differ in at least *4* positions; and
- Each pair of words *x* and *y* in *S* (where *x* and *y* may be identical) are such that <IT>x<SUP>R</SUP></IT> and <IT>y<SUP>C</SUP></IT>
          differ in at least <IT>4</IT> positions.
         Here,
          <IT>( x<SUB>1</SUB>,&#8230;,x<SUB>8</SUB> )<SUP>R</SUP>
              =
              ( x<SUB>8</SUB>,&#8230;,x<SUB>1</SUB> )</IT>
          is the reverse of <IT>( x<SUB>1</SUB>,&#8230;,x<SUB>8</SUB> )</IT> and
          <IT>( y<SUB>1</SUB>,&#8230;,y<SUB>8</SUB> )<SUP>C</SUP></IT>
          is the Watson-Crick complement of <IT>( y<SUB>1</SUB>,&#8230;,y<SUB>8</SUB> )</IT>, i.e.
          the word where
            each <IT>A</IT> is replaced by a <IT>T</IT> and vice versa and
            each <IT>C</IT> is replaced by a <IT>G</IT> and vice versa.