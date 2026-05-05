// Test of H* that contributions will be lower under losses
// mixed model with random intercepts for group and ss
// losses is a dummy with 1=losses, 0 otherwise

mixed cont losses || sessioncode: || id:

// exploratory analysis 
// test that loss aversion and fear of non-contributions are greater under losses

mixed loss_aversion losses || sessioncode: || id:
mixed fear losses || sessioncode: || id:

