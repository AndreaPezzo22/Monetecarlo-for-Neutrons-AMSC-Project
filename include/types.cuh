#ifndef TYPES_H
#define TYPES_H

struct Material {
	float sigma_s;
	float sigma_a;
	float sigma_t;
};

// Aggiunto Struct Region per definire le posizioni dei materiali 
struct Region {
	float min_ix, max_ix;
	float min_iy, max_iy;
	float min_iz, max_iz;
	int mat_id;
};

// Struct Particella
struct Particle {
	float x, y, z;
};

#endif