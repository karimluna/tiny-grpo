package main

import "fmt"

type Metrics struct {
	Load      float64
	LatencyMs float64
	ErrorRate float64
	QueueMs   float64
}

func Reward(m Metrics, alpha, beta float64) float64 {
	return -(m.LatencyMs + alpha*m.ErrorRate*1000 + beta*m.QueueMs)
}

type Backend struct {
	ID      int
	Metrics Metrics
}

type Router struct {
	policy   *Policy
	backends []*Backend
	alpha    float64
	beta     float64
}

func NewRouter(backends []*Backend, lr, eps, alpha, beta float64) *Router {
	// 4 features: load, latency, errorRate, queueMs (see Features() in policy.go)
	return &Router{
		policy:   NewPolicy(len(backends), 4, lr, eps),
		backends: backends,
		alpha:    alpha,
		beta:     beta,
	}
}

// aggregateFeatures builds a single feature vector from the worst-case
// metrics across all backends, tis keeps the policy sensitive to any
// degraded server rather than masking it with an average.
func (r *Router) aggregateFeatures() []float64 {
	var worst Metrics
	for _, b := range r.backends {
		if b.Metrics.Load > worst.Load {
			worst.Load = b.Metrics.Load
		}
		if b.Metrics.LatencyMs > worst.LatencyMs {
			worst.LatencyMs = b.Metrics.LatencyMs
		}
		if b.Metrics.ErrorRate > worst.ErrorRate {
			worst.ErrorRate = b.Metrics.ErrorRate
		}
		if b.Metrics.QueueMs > worst.QueueMs {
			worst.QueueMs = b.Metrics.QueueMs
		}
	}
	return Features(worst)
}

func (r *Router) Route() int {
	x := r.aggregateFeatures()
	a1, a2 := r.policy.SamplePair(x)

	r1 := Reward(r.backends[a1].Metrics, r.alpha, r.beta)
	r2 := Reward(r.backends[a2].Metrics, r.alpha, r.beta)

	mean := (r1 + r2) / 2.0
	r.policy.GRPOUpdate(x, a1, r1-mean)
	r.policy.GRPOUpdate(x, a2, r2-mean)

	return a1
}

func (r *Router) PolicyProbs() []float64 {
	return r.policy.Probs(r.aggregateFeatures())
}

// StateDesc returns a human-readable summary of the current feature vector
func (r *Router) StateDesc() string {
	x := r.aggregateFeatures()
	return fmt.Sprintf("load=%.2f lat=%.2f err=%.2f q=%.2f", x[0], x[1], x[2], x[3])
}
