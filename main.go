package main

import (
	"fmt"
	"math/rand"
	"strings"
)

type SimBackend struct {
	baseLat   float64
	loadSens  float64
	baseError float64
	active    int
	decay     float64
}

func (b *SimBackend) Observe() Metrics {
	load := float64(b.active) / 100.0
	if load > 1 {
		load = 1
	}
	lat := b.baseLat + b.loadSens*float64(b.active)
	err := b.baseError + 0.01*float64(b.active)
	if err > 1 {
		err = 1
	}
	queue := 0.0
	if b.active > 10 {
		queue = float64(b.active-10) * 5.0
	}
	return Metrics{Load: load, LatencyMs: lat, ErrorRate: err, QueueMs: queue}
}

func (b *SimBackend) Hit()  { b.active++ }
func (b *SimBackend) Tick() { b.active = int(float64(b.active) * b.decay) }

func bar(pct float64, width int) string {
	n := int(pct / 100 * float64(width))
	if n < 0 {
		n = 0
	}
	if n > width {
		n = width
	}
	return strings.Repeat("█", n) + strings.Repeat("░", width-n)
}

func main() {
	rand.Seed(42)

	sims := []*SimBackend{
		{baseLat: 20, loadSens: 3.0, baseError: 0.001, decay: 0.92}, // fast & reliable
		{baseLat: 40, loadSens: 2.0, baseError: 0.003, decay: 0.90}, // medium
		{baseLat: 80, loadSens: 1.0, baseError: 0.002, decay: 0.88}, // slow, good under load
		{baseLat: 25, loadSens: 3.5, baseError: 0.050, decay: 0.90}, // fast but flaky
	}

	backends := make([]*Backend, len(sims))
	for i := range sims {
		backends[i] = &Backend{ID: i, Metrics: sims[i].Observe()}
	}

	router := NewRouter(backends, 0.01, 0.10, 1.5, 0.3)

	steps := 3000
	rollingR := make([]float64, 0, 200)

	fmt.Println("╔════════════════════════════════════════════════╗")
	fmt.Println("║     	      Tiny GRPO Load Balancer             ║")
	fmt.Println("╚════════════════════════════════════════════════╝")
	fmt.Printf("servers=%d  K=2  lr=0.01  ε=0.10  α=1.5  β=0.3\n\n", len(sims))

	for step := 1; step <= steps; step++ {
		if step == 1000 {
			sims[0].baseLat = 180
			sims[0].baseError = 0.06
			fmt.Println("\n  Server 0 degraded  (lat 20 -> 180ms, err 0.1% -> 6%)")
		}
		if step == 2000 {
			sims[0].baseLat = 20
			sims[0].baseError = 0.001
			fmt.Println("\n  Server 0 recovered (lat -> 20ms, err -> 0.1%)")
		}

		for i, s := range sims {
			backends[i].Metrics = s.Observe()
		}

		chosen := router.Route()
		sims[chosen].Hit()

		r := Reward(sims[chosen].Observe(), 1.5, 0.3)
		rollingR = append(rollingR, r)
		if len(rollingR) > 200 {
			rollingR = rollingR[1:]
		}

		for _, s := range sims {
			s.Tick()
		}

		if step%500 == 0 {
			probs := router.PolicyProbs()
			avgR := 0.0
			for _, v := range rollingR {
				avgR += v
			}
			avgR /= float64(len(rollingR))

			fmt.Printf("Step %4d | π=[", step)
			for i, p := range probs {
				if i > 0 {
					fmt.Print(" ")
				}
				fmt.Printf("%4.1f%%", p*100)
			}
			fmt.Printf("] avgR=%6.1f | %s\n", avgR, router.StateDesc())
		}
	}

	fmt.Println("\nFinal Policy:")
	probs := router.PolicyProbs()
	for i, p := range probs {
		fmt.Printf("  Server %d: %5.1f%% %s\n", i, p*100, bar(p*100, 30))
	}
}
