export type ProjectStatus = "Planning" | "Under construction" | "Ready to move" | "Delivered";

export type DocumentType =
  | "Project information"
  | "Tower information"
  | "Floor plan"
  | "Unit details"
  | "Amenities"
  | "Pricing"
  | "Legal document"
  | "RERA certificate"
  | "Possession details"
  | "Payment plan"
  | "Brochure"
  | "Other";

export type Builder = {
  id: string;
  name: string;
  address: string;
  contactPerson: string;
  phone: string;
  email: string;
};

export type Project = {
  id: string;
  builderId: string;
  name: string;
  address: string;
  city: string;
  state: string;
  reraNumber: string;
  status: ProjectStatus;
  towers: string[];
};

export type ProjectDocument = {
  id: string;
  builderId: string;
  projectId: string;
  title: string;
  description: string;
  type: DocumentType;
  fileName: string;
  uploadDate: string;
  sizeMb: number;
  tags: string[];
  notes: string;
  relatedTower?: string;
  relatedUnitType?: string;
};
