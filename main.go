package main

import (
	"fmt"
	"strings"
)

// S simulated server with load-dependend behavior
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
		load = 1 // saturated load
	}
	lat := b.baseLat + b.loadSens*float64(b.active)
	err := b.baseError + 0.01*float64(b.active)
	if err > 1 {
		err = 1 // saturated error
	}
	queue := 0.0
	if b.active > 10 {
		queue = float64(b.active-10) * 5.0
	}
	return Metrics{Load: load, LatencyMs: lat, ErrorRate: err, QueueMs: queue}
}

func (b *SimBackend) Hit()  { b.active++ }
func (b *SimBackend) Tick() { b.active = int(float64(b.active) * b.decay) }

func bar(pct float64) string {
	n := int(pct / 2)
	return strings.Repeat("█", n) + strings.Repeat("░", 50-n)
}

func main() {
	sims := []*SimBackend{
		{baseLat: 20, loadSens: 2.0, baseError: 0.001, decay: 0.92},  // fast & reliable
		{baseLat: 60, loadSens: 1.5, baseError: 0.005, decay: 0.90},  // medium
		{baseLat: 150, loadSens: 3.0, baseError: 0.010, decay: 0.88}, // slow
		{baseLat: 30, loadSens: 4.0, baseError: 0.080, decay: 0.90},  // fast but flaky
	}

	backends := make([]*Backend, len(sims))
	for i := range sims {
		backends[i] = &Backend{ID: i, Metrics: sims[i].Observe()}
	}
	router := NewRouter(backends, 0.02, 10.0, 1.0, 2.0, 0.5, 2)

	counts := make([]int, len(sims))
	totalR := 0.0
	steps := 3000
	fmt.Println("╔══════════════════════════════════════╗")
	fmt.Println("║   Tiny GRPO Load Balancer (Go)       ║")
	fmt.Println("╚══════════════════════════════════════╝")
	fmt.Printf("backends=%d  K=2  lr=0.02  α=2.0  β=0.5\n\n", len(sims))

	print := func(step int) {
		state := router.aggregateState()
		probs := router.PolicyProbs(state)
		fmt.Printf("Step %4d | π=[", step)
		for i, p := range probs {
			if i > 0 {
				fmt.Print(" ")
			}
			fmt.Printf("%.2f", p)
		}
		fmt.Printf("] routes=[")
		for i, c := range counts {
			if i > 0 {
				fmt.Print(" ")
			}
			fmt.Printf("%d%%", c*100/step)
		}
		fmt.Printf("] avgR=%.1f state=%s\n", totalR/float64(step), state)
	}
	for step := 1; step <= steps; step++ {
		// Phase shifts to demonstrate adaptation
		if step == 1000 {
			sims[0].baseLat = 180
			sims[0].baseError = 0.06
			fmt.Println("\n Server 0 degraded  (lat 20 -> 180ms, err 0.1% -> 6%)")
		}
		if step == 2000 {
			sims[0].baseLat = 20
			sims[0].baseError = 0.001
			fmt.Println("\n Server 0 recovered (lat -> 20ms, err -> 0.1%)")
		}

		for i, s := range sims {
			backends[i].Metrics = s.Observe()
		}

		chosen := router.Route()
		sims[chosen].Hit()

		totalR += Reward(sims[chosen].Observe(), 2.0, 0.5)
		counts[chosen]++

		for _, s := range sims {
			s.Tick()
		}

		if step%500 == 0 {
			print(step)
		}
	}
	fmt.Println("\n┌──────────────────────────────────────────────────────────┐")
	fmt.Println("│ Final Policy                                             │")
	fmt.Println("├──────────────────────────────────────────────────────────┤")
	state := router.aggregateState()
	probs := router.PolicyProbs(state)
	for i, p := range probs {
		fmt.Printf("│ Server %d: %5.1f%% %s │\n", i, p*100, bar(p*100))
	}
	fmt.Println("└──────────────────────────────────────────────────────────┘")
}
