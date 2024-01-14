%facts of couples
couple('George','Mum').
couple('Spencer','Kydd').
couple('Elizabeth','Philip').
couple('Diana','Charles').
couple('Anne','Mark').
couple('Andrew','Sarah').
couple('Edward','Sophie').

%facts of gender
male('George').
male('Philip').
male('Spencer').
male('Charles').
male('Mark').
male('Andrew').
male('Edward').
male('William').
male('Harry').
male('Peter').
male('James').

female('Mum').
female('Kydd').
female('Elizabeth').
female('Margaret').
female('Diana').
female('Anne').
female('Sarah').
female('Sophie').
female('Zara').
female('Beatrice').
female('Eugenie').
female('Louise').

%father(X,Y)--X is Y's father
father('George','Elizabeth').
father('George','Margaret').
father('Spencer','Diana').
father('Philip','Charles').
father('Philip','Anne').
father('Philip','Andrew').
father('Philip','Edward').
father('Charles','William').
father('Charles','Harry').
father('Mark','Peter').
father('Mark','Zara').
father('Andrew','Beatrice').
father('Andrew','Eugenie').
father('Edward','Louise').
father('Edward','James').

%mother(X,Y)--X is Y's mother
mother(X,Z) :- father(Y,Z),(couple(X,Y)|couple(Y,X)).

%facts of children;child(X,Y)--X is Y's child
child_fact('Elizabeth',couple('George','Mum')).
child_fact('Margaret',couple('George','Mum')).
child_fact('Diana',couple('Spencer','Kydd')).
child_fact('Charles',couple('Elizabeth','Philip')).
child_fact('Anne',couple('Elizabeth','Philip')).
child_fact('Andrew',couple('Elizabeth','Philip')).
child_fact('Edward',couple('Elizabeth','Philip')).
child_fact('William',couple('Diana','Charles')).
child_fact('Harry',couple('Diana','Charles')).
child_fact('Peter',couple('Anne','Mark')).
child_fact('Zara',couple('Anne','Mark')).
child_fact('Beatrice',couple('Andrew','Sarah')).
child_fact('Eugenie',couple('Andrew','Sarah')).
child_fact('Louise',couple('Edward','Sophie')).
child_fact('James',couple('Edward','Sophie')).

child(X,Y) :- parent(Y,X).

list_mother :- mother(X,Y), write(X),write(' is '),write(Y),write("'s mother"),nl,fail.
find_mother :- read(X),mother(Y,X),write(Y),fail.
%judge_mother :- read(X),read(Y),mother(X,Y),fail.

%Son
son(X,Y) :- (father(Y,X)|mother(Y,X)),male(X).
list_son :- son(X,Y), write(X),write(' is '),write(Y),write("'s son"),nl,fail.
find_son :- read(X),son(Y,X),write(Y),nl,fail.

%Daughter
daughter(X,Y) :- (father(Y,X)|mother(Y,X)),female(X).
list_daughter :- daughter(X,Y), write(X),write(' is '),write(Y),write("'s daughter"),nl,fail.
find_daughter :- read(X),daughter(Y,X),write(Y),nl,fail.

%Sister
sister(X,Y) :- father(Z,X),father(Z,Y),X\=Y,female(X).
list_sister :- sister(X,Y), write(X),write(' is '),write(Y),write("'s sister"),nl,fail.
find_sister :- read(X),sister(Y,X),write(Y),nl,fail.

%Brother
brother(X,Y) :- father(Z,X),father(Z,Y),X\=Y,male(X). 

%parent
parent(X,Y) :- father(X,Y)|mother(X,Y).

%Elizabeth's grandchildren;Diana's brother-in-law;Zara's greatgrandparents;Eugenie's ancestors
%grandchild(X,Y) :- (father(Y,Z),father(Z,X))|(father(Y,Z),mother(Z,X))|(mother(Y,Z),father(Z,X))|(mother(Y,Z),mother(Z,X)).

grandchild(X,Y) :- parent(Y,Z),parent(Z,X).
list_Elizabeth_grandchildren :- grandchild(X,'Elizabeth'),write(X),write(" is Elizabeth's grandchild"),nl,fail.

brother_in_law(X,Y) :- brother(X,Z),(couple(Z,Y)|couple(Y,Z)) | (couple(X,Z)|couple(Z,X)),sister(Z,Y) | (couple(Z,Y)|couple(Y,Z)),sister(W,Z),(couple(X,W)|couple(W,X)).
list_Diana_brother_in_law :- brother_in_law(X,'Diana'),write(X),write(" is Diana's brother-in-law"),nl,fail.

greatgrandparent(X,Y) :- parent(X,Z),parent(Z,W),parent(W,Y).
list_Zara_grandparents :- greatgrandparent(X,'Zara'),write(X),write(" is Zara's greatgrandparent"),nl,fail.

ancestor(X,Y) :- parent(X,Y)|grandchild(Y,X)|greatgrandparent(X,Y).
list_Eugenie_ancestor :- ancestor(X,'Eugenie'),write(X),write(" is Eugenie's ancestor"),nl,fail.

%mth cousin n times removed
distance(X, Y, N) :- (X = Y, N = 0);
(ancestor(X, Y), child(Y, Z), distance(X, Z, N1), N is N1 + 1);
(\+ancestor(X, Y), ancestor(Z, X), ancestor(Z, Y), distance(Z, Y, N1), distance(Z, X, N2), N is (N1 - N2)).

mthCousin(X, Y, M) :- (ancestor(Z, X), ancestor(Z, Y), distance(Z, X, N1), distance(Z, Y, N2), N1 = N2, M is (N1 - 1)).

mthCousinNremoved(X, Y, M, N) :- ((mthCousin(X, Y, M1), M is M1, N is 0)|(ancestor(Z, X), mthCousin(Z, Y, M), distance(Z, X, N2), N is N2)|
(ancestor(Z, Y), mthCousin(Z, X, M), distance(Z, Y, N3), N is N3)). 
