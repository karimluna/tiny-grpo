package main

import (
	"math"
	"math/rand"
)

// policy is a tabular softmax, no neural networks just a map
type Policy struct {
	w        map[string][]float64
	nActions int
	lr       float64
	clip     float64
	temp     float64
}

func NewPolicy(nActions int, lr, clip, temp float64) *Policy {
	return &Policy{
		w:        make(map[string][]float64),
		nActions: nActions,
		lr:       lr,
		clip:     clip,
		temp:     temp,
	}
}

func (p *Policy) ensure(key string) {
	if _, ok := p.w[key]; !ok {
		p.w[key] = make([]float64, p.nActions)
	}
}

// probs returns the sofrmax probability vector for a state
func (p *Policy) Probs(key string) []float64 {
	p.ensure(key)
	w := p.w[key]
	out := make([]float64, p.nActions)

	mx := w[0] // stable softmax
	for _, v := range w {
		if v > mx {
			mx = v
		}
	}
	sum := 0.0
	for i, v := range w {
		out[i] = math.Exp((v - mx) / p.temp)
		sum += out[i]
	}

	for i := range out {
		out[i] /= sum
	}
	return out
}

// sample draws an action from policy
func (p *Policy) Sample(key string) int {
	pr := p.Probs(key)
	r := rand.Float64()
	cum := 0.0
	for i, v := range pr {
		cum += v
		if r < cum {
			return i
		}
	}
	return p.nActions - 1
}

// GRPOUpdate applies one policy-gradient step
//
//	∇_w log π(a|s) = 1(a'=a) - π(a'|s)     (softmax gradient)
//
//	w[s][a'] += lr · clip(A, -C, +C) · (1(a'=a) - π(a'|s))
func (p *Policy) GRPOUpdate(key string, action int, advantage float64) {
	p.ensure(key)
	pr := p.Probs(key)

	adv := math.Max(-p.clip, math.Min(p.clip, advantage))

	for a := 0; a < p.nActions; a++ {
		grad := -pr[a]
		if a == action {
			grad = 1.0 - pr[a]
		}
		p.w[key][a] += p.lr * adv * grad
	}
}
