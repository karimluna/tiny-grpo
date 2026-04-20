package main

import (
	"math"
	"math/rand"
)

// Policy: linear logit model with epsilon-greedy exploration.
type Policy struct {
	// w[action][feature], 4 actions × 4 features = 16 floats total
	w        [][]float64
	nActions int
	nFeats   int
	lr       float64
	eps      float64
}

func NewPolicy(nActions, nFeats int, lr, eps float64) *Policy {
	w := make([][]float64, nActions)
	for a := range w {
		w[a] = make([]float64, nFeats)
	}
	return &Policy{
		w:        w,
		nActions: nActions,
		nFeats:   nFeats,
		lr:       lr,
		eps:      eps,
	}
}

// Features maps raw metrics into a bounded [0,1] feature vector.
// Normalisation constants are soft ceilings, not hard ones,
// values above them just saturate toward 1, which is fine.
func Features(m Metrics) []float64 {
	return []float64{
		clamp01(m.Load),
		clamp01(m.LatencyMs / 300.0),
		clamp01(m.ErrorRate / 0.10),
		clamp01(m.QueueMs / 500.0),
	}
}

func clamp01(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

// logits computes w[a]·x for each action.
func (p *Policy) logits(x []float64) []float64 {
	out := make([]float64, p.nActions)
	for a := range out {
		for f, xf := range x {
			out[a] += p.w[a][f] * xf
		}
	}
	return out
}

// softmax over logit slice (numerically stable).
func softmax(logits []float64) []float64 {
	mx := logits[0]
	for _, v := range logits {
		if v > mx {
			mx = v
		}
	}
	out := make([]float64, len(logits))
	sum := 0.0
	for i, v := range logits {
		out[i] = math.Exp(v - mx)
		sum += out[i]
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

// Probs returns the epsilon-mixed action probabilities given a feature vector.
func (p *Policy) Probs(x []float64) []float64 {
	pr := softmax(p.logits(x))
	for i := range pr {
		pr[i] = (1-p.eps)*pr[i] + p.eps/float64(p.nActions)
	}
	return pr
}

func sampleCat(pr []float64) int {
	r := rand.Float64()
	cum := 0.0
	for i, v := range pr {
		cum += v
		if r < cum {
			return i
		}
	}
	return len(pr) - 1
}

func (p *Policy) Sample(x []float64) int {
	return sampleCat(p.Probs(x))
}

// SamplePair draws two distinct actions for GRPO group comparison.
func (p *Policy) SamplePair(x []float64) (int, int) {
	pr := p.Probs(x)
	a1 := sampleCat(pr)

	// Sample a2 from the remaining mass, no renormalisation loop needed.
	total := 1.0 - pr[a1]
	r := rand.Float64() * total
	cum := 0.0
	a2 := (a1 + 1) % p.nActions
	for i, v := range pr {
		if i == a1 {
			continue
		}
		cum += v
		if r < cum {
			a2 = i
			break
		}
	}
	return a1, a2
}

// GRPOUpdate performs a REINFORCE-style policy gradient step.
//
// The gradient of the expected reward w.r.t. w[a][f] is:
//
//	∂/∂w[a][f] = adv · (1{a==chosen} - π(a)) · x[f]
func (p *Policy) GRPOUpdate(x []float64, action int, advantage float64) {
	pr := softmax(p.logits(x)) // use raw softmax, not eps-mixed, for gradient
	for a := 0; a < p.nActions; a++ {
		indicator := 0.0
		if a == action {
			indicator = 1.0
		}
		grad := advantage * (indicator - pr[a])
		for f, xf := range x {
			p.w[a][f] += p.lr * grad * xf
		}
	}
}
