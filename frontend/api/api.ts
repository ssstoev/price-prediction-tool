const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export interface criteriaParams {
    size_m2: number;
    nr_of_rooms: number;
    floor: number;
    building_total_floors: number;
    neighbourhood: string,
    is_first_floor: number,
    is_last_floor: number,
    is_furnished: number,
    near_public_transport: number

}

export interface PredictionResponse {
    lower_bound: number;
    upper_bound: number;
}

export async function callInferenceModel(criteria: criteriaParams): Promise<PredictionResponse> {
    console.log("Calling API...")
    const response = await fetch(`${API_BASE}/predictPricePerSqm`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(criteria)
    });

    if (!response.ok) throw new Error("Search failed");

    const data = await response.json();
    const totalPrice = data.normalized_result * criteria.size_m2;

    return {
        lower_bound: Math.round(totalPrice * 0.90),
        upper_bound: Math.round(totalPrice * 1.10)
    };
}