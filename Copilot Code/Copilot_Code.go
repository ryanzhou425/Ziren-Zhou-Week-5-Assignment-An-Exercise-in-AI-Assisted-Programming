package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"strconv"
	"sync"

	"github.com/montanaflynn/stats"
)

type Record struct {
	Dataset string
	X       float64
	Y       float64
}

// Group data by dataset name
func groupByDataset(data []Record) map[string][]Record {
	groups := make(map[string][]Record, len(data))
	for _, d := range data {
		groups[d.Dataset] = append(groups[d.Dataset], d)
	}
	return groups
}

// Simple linear regression function
func simpleLinearRegression(records []Record) (float64, float64, error) {
	var xVals, yVals []float64
	for _, r := range records {
		xVals = append(xVals, r.X)
		yVals = append(yVals, r.Y)
	}

	xMean, err := stats.Mean(xVals)
	if err != nil {
		return 0, 0, err
	}
	yMean, err := stats.Mean(yVals)
	if err != nil {
		return 0, 0, err
	}

	numerator := 0.0
	denominator := 0.0
	for i := range xVals {
		numerator += (xVals[i] - xMean) * (yVals[i] - yMean)
		denominator += (xVals[i] - xMean) * (xVals[i] - xMean)
	}
	slope := numerator / denominator
	intercept := yMean - slope*xMean

	return slope, intercept, nil
}

func mean(values []float64) float64 {
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

func readCSV(fileName string) ([]Record, error) {
	file, err := os.Open(fileName)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	var data []Record
	for i, row := range records {
		if i == 0 {
			continue // skip header
		}
		x, err := strconv.ParseFloat(row[1], 64)
		if err != nil {
			return nil, fmt.Errorf("error parsing X value in row %d: %v", i, err)
		}
		y, err := strconv.ParseFloat(row[2], 64)
		if err != nil {
			return nil, fmt.Errorf("error parsing Y value in row %d: %v", i, err)
		}
		data = append(data, Record{Dataset: row[0], X: x, Y: y})
	}
	return data, nil
}

func main() {
	const fileName = "anscombe.csv"
	data, err := readCSV(fileName)
	if err != nil {
		log.Fatal(err)
	}

	// Group data
	groups := groupByDataset(data)

	// Perform regression by group
	var wg sync.WaitGroup
	for datasetName, records := range groups {
		wg.Add(1)
		go func(name string, recs []Record) {
			defer wg.Done()
			slope, intercept, err := simpleLinearRegression(recs)
			if err != nil {
				log.Printf("Error processing dataset %s: %v", name, err)
				return
			}
			fmt.Printf("Dataset: %s | Slope: %.4f | Intercept: %.4f\n", name, slope, intercept)
		}(datasetName, records)
	}
	wg.Wait()
}
