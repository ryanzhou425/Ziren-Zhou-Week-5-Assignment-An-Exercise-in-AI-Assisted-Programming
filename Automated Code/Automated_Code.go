package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"strconv"

	"github.com/montanaflynn/stats"
)

type Record struct {
	Dataset string
	X       float64
	Y       float64
}

// Group data by dataset name
func groupByDataset(data []Record) map[string][]Record {
	groups := make(map[string][]Record)
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

func main() {
	// Open the CSV file
	file, err := os.Open("anscombe.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	// Read CSV contents
	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	// Skip header and process from the second row
	var data []Record
	for i, row := range records {
		if i == 0 {
			continue // skip header
		}
		x, _ := strconv.ParseFloat(row[1], 64)
		y, _ := strconv.ParseFloat(row[2], 64)
		rec := Record{
			Dataset: row[0],
			X:       x,
			Y:       y,
		}
		data = append(data, rec)
	}

	// Group data
	groups := groupByDataset(data)

	// Perform regression by group
	for datasetName, records := range groups {
		slope, intercept, err := simpleLinearRegression(records)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("Dataset: %s | Slope: %.4f | Intercept: %.4f\n", datasetName, slope, intercept)
	}
}
