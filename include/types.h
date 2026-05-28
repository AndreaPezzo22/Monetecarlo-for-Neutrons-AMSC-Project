#ifndef TYPES_H
#define TYPES_H

struct Material {
	float sigma_s;
	float sigma_a;
	float sigma_t;
};

// Aggiunto Struct Region per definire le posizioni dei materiali 
struct Region {
	float min_x, max_x;
	float min_y, max_y;
	float min_z, max_z;
	u_int8_t mat_id;
};

#endif