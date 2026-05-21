#ifndef MATERIALS_H
#define MATERIALS_H

// Detects the material where the particle is located based on its position. 
// This function checks the position against defined regions and returns 
// the corresponding material ID.
//
// Parameters:
// - pos: the current position of the particle in 3D space.
// - final_mat_id: the material ID to return if the position is not inside any defined 
//   region.
// - inside: a boolean flag indicating whether the position is inside the current region being checked.
// - r: current region being checked for the particle's position.
// - c_num_regions: total number of defined regions to check against.
// - c_regions: array of defined regions with their corresponding material IDs.
// - mat_id: the material ID to return if the position is inside the current region being checked.
// 
// Preconditions:
// - 
//
// Behavior:
// At each iteration, the function checks if the particle's position is inside any of the defined regions.
// If it is inside a region, it updates the final material ID to the one corresponding to that region. 
// If it is not inside any region, it returns the default material ID (Vacuum).

#include "types.h"

extern __constant__ Region c_regions[]; 
extern __constant__ int c_num_regions; 

__device__ inline u_int8_t getMaterialID(float3 pos) {

	u_int8_t final_mat_id = 0;
	
	for ( int i=0; i<c_num_regions; i++) {
		Region r = c_regions[i];
		int inside = (pos.x >= r.min_ix) * (pos.x <= r.max_ix) *
			     (pos.y >= r.min_iy) * (pos.y <= r.max_iy) *
			     (pos.z >= r.min_iz) * (pos.z <= r.max_iz);
		final_mat_id = (inside * r.mat_id) + ((1 - inside) * final_mat_id);
	}
	return final_mat_id;
}
	
#endif

/*
	Nel readme del progetto va specificato che per evitare problemi nella selezione del materiale.
	La funzione fa un return dell ultimo materiale trovato, se il punto è dentro più regioni. 
	Quindi è importante che le regioni siano definite in ordine di priorità, con le regioni più specifiche 
	(ad esempio, un oggetto all'interno di un altro) definite dopo quelle più generali (ad esempio, il vuoto circostante). 
	In questo modo, se un punto è all'interno di più regioni, verrà restituito il materiale della regione più specifica.

	La soluzione eventuale è quella di segmentare i confini tra le regioni, in modo che non ci siano sovrapposizioni.
	Senza il bisogno di segmentare la regione di base (vacuum) che è sempre presente, ma segmentare le regioni più 
	specifiche in modo che non si sovrappongano tra loro.
*/