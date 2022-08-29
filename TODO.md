
fix backprop :
For adversial it needs to use P(s,a) = t * P(s,a) - (1-t) * max( P(s+1) )
a = action, s = state, s+1 = next state, t = time discount factor

try bootstrapping with MC(50)

