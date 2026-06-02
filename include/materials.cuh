/**
 * @brief Detects the material containing a given particle position.
 *
 * @param pos Current position of the particle in 3D space.
 *
 * @details
 *  - Iterates through the device constant region array `c_regions`.
 *  - If `pos` is inside a region, the function updates the material ID to that region's `mat_id`.
 *  - If `pos` is not inside any defined region, the function returns the default material ID 0.
 *
 * @note The function returns the last matching region's material ID when multiple regions overlap.
 *       Therefore, regions should be ordered from general to specific, with the most specific
 *       regions defined last.
 */

#ifndef MATERIALS_H
#define MATERIALS_H

#include "types.h"

extern __constant__ Region c_regions[]; 
extern __constant__ int c_num_regions; 

__device__ inline u_int8_t getMaterialID(float3 pos) {

	u_int8_t final_mat_id = 0;
	
	for ( int i=0; i<c_num_regions; i++) {
		Region r = c_regions[i];
		int inside = (pos.x >= r.min_x) * (pos.x <= r.max_x) *
			     (pos.y >= r.min_y) * (pos.y <= r.max_y) *
			     (pos.z >= r.min_z) * (pos.z <= r.max_z);
		final_mat_id = (inside * r.mat_id) + ((1 - inside) * final_mat_id);
	}
	// printf("Position: (%.2f, %.2f, %.2f) -> Material ID: %d\n", pos.x, pos.y, pos.z, final_mat_id);
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