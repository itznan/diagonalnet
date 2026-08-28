package main

import (
	"sync"
)

// ReduceParameterGradients accumulates gradients from multiple workers into a master parameter
// using lock-free contiguous chunk partitioning across worker Goroutines.
func ReduceParameterGradients(master *Parameter, workers []*Parameter, numWorkers int) {
	if master == nil || len(workers) == 0 {
		return
	}
	L := len(master.Grad)
	if L == 0 {
		return
	}

	if numWorkers <= 0 {
		numWorkers = len(workers)
	}
	if numWorkers > L {
		numWorkers = L
	}

	chunkSize := (L + numWorkers - 1) / numWorkers
	var wg sync.WaitGroup

	for w := 0; w < numWorkers; w++ {
		start := w * chunkSize
		end := start + chunkSize
		if end > L {
			end = L
		}
		if start >= end {
			continue
		}

		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			for i := s; i < e; i++ {
				var sum float32
				for k := 0; k < len(workers); k++ {
					sum += workers[k].Grad[i]
				}
				master.Grad[i] = sum
			}
		}(start, end)
	}

	wg.Wait()
}

// ReduceGradients accumulates gradients across an entire parameter list for multiple worker replicas.
func ReduceGradients(masterParams []*Parameter, workerParams [][]*Parameter, numWorkers int) {
	if len(masterParams) == 0 || len(workerParams) == 0 {
		return
	}
	for paramIdx, master := range masterParams {
		workersForParam := make([]*Parameter, len(workerParams))
		for k := 0; k < len(workerParams); k++ {
			if paramIdx < len(workerParams[k]) {
				workersForParam[k] = workerParams[k][paramIdx]
			}
		}
		ReduceParameterGradients(master, workersForParam, numWorkers)
	}
}
