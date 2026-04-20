package main

import "fmt"

// metrics holds real-time observations for a backend
type Metrics struct {
	Load      float64 // 0-1 <- normalized
	LatencyMs float64
	ErrorRate float64 // 0-1
	QueueMs   float64
}

// reward function (here is the magic he)
// r = -(latency + α·error_rate·1000 + β·queue_ms) <- Higher is better
func Reward(m Metrics, alpha, beta float64) float64 {
	return -(m.LatencyMs + alpha*m.ErrorRate*1000 + beta*m.QueueMs)
} // ref: http://www.yndxxb.ynu.edu.cn/yndxxbzrkxb/article/doi/10.7540/j.ynu.20240371?viewType=HTML

// here we discretize aggregate metrics into a hashable key for the
// tabular policy. Each metric is {low, mid, high} ->
func StateKey(m Metrics) string {
	lb := min(int(m.Load*3), 2)
	el := 0
	if m.LatencyMs > 100 {
		el = 1
	}
	if m.LatencyMs > 300 {
		el = 2
	}
	er := 0
	if m.ErrorRate > 0.01 {
		er = 1
	}
	if m.ErrorRate > 0.05 {
		er = 2
	}
	return fmt.Sprintf("%d%d%d", lb, el, er)
}

// backend is a routing agent
type Backend struct {
	ID      int
	Metrics Metrics
}

// ruter applies GRPO to load-balancing decisions.
type Router struct {
	policy   *Policy
	backends []*Backend
	alpha    float64
	beta     float64
	K        int
}

func NewRouter(backends []*Backend, lr, clip, temp, alpha, beta float64, K int) *Router {
	return &Router{
		policy:   NewPolicy(len(backends), lr, clip, temp),
		backends: backends,
		alpha:    alpha,
		beta:     beta,
		K:        K,
	}
}

// GRPO loop: sample K actions, observe rewards, compute
// group, advantages, update policy, return first sample
func (r *Router) Route() int {
	state := r.aggregateState()

	// sample K actions
	actions := make([]int, r.K)
	for i := 0; i < r.K; i++ {
		actions[i] = r.policy.Sample(state)
	}

	// observe reward for each sampled action
	rewards := make([]float64, r.K)
	for i, a := range actions {
		rewards[i] = Reward(r.backends[a].Metrics, r.alpha, r.beta)
	}

	// group advantage: A_i = r_i - mean(r)
	mean := 0.0
	for _, rw := range rewards {
		mean += rw
	}
	mean /= float64(r.K)

	// update policy
	for i, a := range actions {
		r.policy.GRPOUpdate(state, a, rewards[i]-mean)
	}

	return actions[0]
}

func (r *Router) aggregateState() string {
	var avg Metrics
	n := float64(len(r.backends))
	for _, b := range r.backends {
		avg.Load += b.Metrics.Load / n
		avg.LatencyMs += b.Metrics.LatencyMs / n
		avg.ErrorRate += b.Metrics.ErrorRate / n
		avg.QueueMs += b.Metrics.QueueMs / n
	}
	return StateKey(avg)
}

func (r *Router) PolicyProbs(state string) []float64 {
	return r.policy.Probs(state)
}
