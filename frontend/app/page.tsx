'use client';

import { callInferenceModel } from '@/api/api';
import { useState, FormEvent } from 'react';

// ============================================
// Type Definitions
// ============================================

interface FormState {
  size: string;
  rooms: string; 
  floor: string;
  totalFloors: string;
  neighbourhood: string;
  furnished: boolean;
  nearTransport: boolean;
}

interface PredictionResult {
  price_low: number;
  price_high: number;
}

interface FieldProps {
  label: string;
  name: keyof FormState;
  type: 'number' | 'select';
  value: string;
  onChange: (name: keyof FormState, value: string) => void;
  options?: string[];
  placeholder?: string;
  error?: string;
  selectPlaceholder?: string;
}

interface ToggleProps {
  label: string;
  name: keyof FormState;
  checked: boolean;
  onChange: (name: keyof FormState, value: boolean) => void;
}

interface ResultCardProps {
  result: PredictionResult;
  visible: boolean;
  t: Translations;
}

type Language = 'en' | 'bg';

interface Translations {
  header: string;
  subtitle: string;
  size: string;
  rooms: string;
  floor: string;
  totalFloors: string;
  neighbourhood: string;
  furnished: string;
  nearTransport: string;
  submit: string;
  calculating: string;
  required: string;
  confidence: string;
  conservative: string;
  optimistic: string;
  select: string;
  neighbourhoods: string[];
}

const translations: Record<Language, Translations> = {
  en: {
    header: 'Property Valuation',
    subtitle: "Estimate Your Property's Value",
    size: 'Size (m²)',
    rooms: 'Number of Rooms',
    floor: 'Apartment Floor',
    totalFloors: 'Total Floors in Building',
    neighbourhood: 'Neighbourhood',
    furnished: 'Furnished',
    nearTransport: 'Near Public Transport',
    submit: 'Get Valuation',
    calculating: 'Calculating...',
    required: 'Required',
    confidence: 'Based on current market data · High confidence',
    conservative: 'Conservative estimate',
    optimistic: 'Optimistic estimate',
    select: 'Select...',
    neighbourhoods: [
      'Надежда 1',
      'Надежда 2',
      'Надежда 3',
      'Надежда 4',
      'Люлин 1',
      'Люлин 2',
      'Люлин 3',
      'Люлин 4',
      'Люлин 5',
      'Люлин 6',
      'Люлин 7',
      'Люлин 8',
      'Люлин 9',
      'Люлин 10',
      'Младост 1',
      'Младост 1А',
      'Младост 2',
      'Младост 3',
      'Младост 4',
      'Дружба 1',
      'Дружба 2',
      'Студентски град',
      'Витоша',
      'Лозенец',
      'Център',
      'Борово',
      'Овча купел',
      'Банишора',
      'Сердика',
      'Илинден',
      'Красна поляна',
      'Красно село',
      'Слатина',
      'Гео Милев',
      'Редута',
      'Подуяне',
      'Хаджи Димитър',
      'Левски',
      'Триадица',
      'Изгрев',
      'Изток',
      'Яворов',
      'Оборище',
      'Докторски паметник',
      'Манастирски ливади',
      'Бъкстон',
      'Павлово',
      'Симеоново',
      'Драгалевци',
      'Бояна',
    ],
  },
  bg: {
    header: 'Оценка на имот',
    subtitle: 'Изчислете стойността на вашия имот',
    size: 'Площ (м²)',
    rooms: 'Брой стаи',
    floor: 'Етаж на апартамента',
    totalFloors: 'Общо етажи в сградата',
    neighbourhood: 'Квартал',
    furnished: 'Обзаведен',
    nearTransport: 'Близо до транспорт',
    submit: 'Изчисли оценка',
    calculating: 'Изчисляване...',
    required: 'Задължително',
    confidence: 'Базирано на актуални пазарни данни · Висока точност',
    conservative: 'Консервативна оценка',
    optimistic: 'Оптимистична оценка',
    select: 'Избери...',
    neighbourhoods: [
      'Надежда 1',
      'Надежда 2',
      'Надежда 3',
      'Надежда 4',
      'Люлин 1',
      'Люлин 2',
      'Люлин 3',
      'Люлин 4',
      'Люлин 5',
      'Люлин 6',
      'Люлин 7',
      'Люлин 8',
      'Люлин 9',
      'Люлин 10',
      'Младост 1',
      'Младост 1А',
      'Младост 2',
      'Младост 3',
      'Младост 4',
      'Дружба 1',
      'Дружба 2',
      'Студентски град',
      'Витоша',
      'Лозенец',
      'Център',
      'Борово',
      'Овча купел',
      'Банишора',
      'Сердика',
      'Илинден',
      'Красна поляна',
      'Красно село',
      'Слатина',
      'Гео Милев',
      'Редута',
      'Подуяне',
      'Хаджи Димитър',
      'Левски',
      'Триадица',
      'Изгрев',
      'Изток',
      'Яворов',
      'Оборище',
      'Докторски паметник',
      'Манастирски ливади',
      'Бъкстон',
      'Павлово',
      'Симеоново',
      'Драгалевци',
      'Бояна',
    ],
  },
};

// ============================================
// Field Component
// ============================================

function Field({ label, name, type, value, onChange, options, placeholder, error, selectPlaceholder = 'Select...' }: FieldProps) {
  const baseInputClasses = `
    w-full px-4 py-3 
    bg-white text-[#1C1C1E] 
    border border-[#E0DCD7] 
    rounded-md
    font-sans text-sm
    transition-colors duration-200
    focus:outline-none focus:border-[#7D8B7A]
    placeholder:text-[#9A9A9C]
  `;

  return (
    <div className="flex flex-col gap-2">
      <label htmlFor={name} className="text-sm text-[#6B6B6D] font-medium">
        {label}
      </label>
      {type === 'select' ? (
        <select
          id={name}
          name={name}
          value={value}
          onChange={(e) => onChange(name, e.target.value)}
          className={`${baseInputClasses} cursor-pointer appearance-none bg-[url('data:image/svg+xml;charset=UTF-8,%3csvg%20xmlns%3d%22http%3a%2f%2fwww.w3.org%2f2000%2fsvg%22%20width%3d%2212%22%20height%3d%2212%22%20viewBox%3d%220%200%2012%2012%22%3e%3cpath%20fill%3d%22%236B6B6D%22%20d%3d%22M2%204l4%204%204-4%22%2f%3e%3c%2fsvg%3e')] bg-[length:12px] bg-[right_16px_center] bg-no-repeat`}
        >
          <option value="">{selectPlaceholder}</option>
          {options?.map((opt) => (
            <option key={opt} value={opt}>
              {opt}
            </option>
          ))}
        </select>
      ) : (
        <input
          id={name}
          name={name}
          type="number"
          value={value}
          onChange={(e) => onChange(name, e.target.value)}
          placeholder={placeholder}
          className={baseInputClasses}
        />
      )}
      {error && <span className="text-xs text-[#DC3545]">{error}</span>}
    </div>
  );
}

// ============================================
// Toggle Component
// ============================================

function Toggle({ label, name, checked, onChange }: ToggleProps) {
  return (
    <div className="flex items-center justify-between py-3">
      <span className="text-sm text-[#6B6B6D] font-medium">{label}</span>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(name, !checked)}
        className={`
          relative w-12 h-6 rounded-full transition-colors duration-200 ease-in-out
          ${checked ? 'bg-[#7D8B7A]' : 'bg-[#E0DCD7]'}
        `}
      >
        <span
          className={`
            absolute top-1 left-1 w-4 h-4 bg-white rounded-full 
            transition-transform duration-200 ease-in-out
            ${checked ? 'translate-x-6' : 'translate-x-0'}
          `}
        />
      </button>
    </div>
  );
}

// ============================================
// Result Card Component
// ============================================

function ResultCard({ result, visible, t }: ResultCardProps) {
  const formatPrice = (price: number): string => {
    return new Intl.NumberFormat('de-DE').format(price);
  };

  return (
    <div
      className={`
        mt-8 pt-8 border-t border-[#E0DCD7]
        transition-all duration-[400ms] ease-out
        ${visible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4 pointer-events-none'}
      `}
    >
      {/* Main Price Range */}
      <div className="text-center mb-6">
        <p className="font-sans text-3xl md:text-4xl font-bold text-[#1C1C1E] tracking-tight">
          {formatPrice(result.price_low)} — {formatPrice(result.price_high)}
        </p>
      </div>

      {/* Confidence Label */}
      <p className="text-center text-sm text-[#6B6B6D] mb-8">
        {t.confidence}
      </p>

      {/* Breakdown Row */}
      <div className="grid grid-cols-2 gap-4">
        <div className="text-center p-4 bg-[#F5F3F0] rounded-md">
          <p className="text-xs text-[#6B6B6D] uppercase tracking-wider mb-1">
            {t.conservative}
          </p>
          <p className="font-sans text-xl font-bold text-[#1C1C1E]">
            {formatPrice(result.price_low)}
          </p>
        </div>
        <div className="text-center p-4 bg-[#F5F3F0] rounded-md">
          <p className="text-xs text-[#6B6B6D] uppercase tracking-wider mb-1">
            {t.optimistic}
          </p>
          <p className="font-sans text-xl font-bold text-[#1C1C1E]">
            {formatPrice(result.price_high)}
          </p>
        </div>
      </div>
    </div>
  );
}

// ============================================
// Language Switcher Component
// ============================================

function LanguageSwitcher({ language, onChange }: { language: Language; onChange: (lang: Language) => void }) {
  return (
    <div className="flex items-center gap-1 bg-[#F5F3F0] rounded-full p-1">
      <button
        type="button"
        onClick={() => onChange('en')}
        className={`
          px-3 py-1.5 text-xs font-medium rounded-full transition-colors duration-200
          ${language === 'en' ? 'bg-white text-[#1C1C1E] shadow-sm' : 'text-[#6B6B6D] hover:text-[#1C1C1E]'}
        `}
      >
        EN
      </button>
      <button
        type="button"
        onClick={() => onChange('bg')}
        className={`
          px-3 py-1.5 text-xs font-medium rounded-full transition-colors duration-200
          ${language === 'bg' ? 'bg-white text-[#1C1C1E] shadow-sm' : 'text-[#6B6B6D] hover:text-[#1C1C1E]'}
        `}
      >
        BG
      </button>
    </div>
  );
}

// ============================================
// Main Component
// ============================================

export default function PropertyValuation() {
  const [language, setLanguage] = useState<Language>('en');
  const t = translations[language];

  const [formState, setFormState] = useState<FormState>({
    size: '',
    rooms: '',
    floor: '',
    totalFloors: '',
    neighbourhood: '',
    furnished: false,
    nearTransport: false,
  });

  const [errors, setErrors] = useState<Partial<Record<keyof FormState, string>>>({});
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [showResult, setShowResult] = useState(false);

  const handleFieldChange = (name: keyof FormState, value: string) => {
    setFormState((prev) => ({ ...prev, [name]: value }));
    if (errors[name]) {
      setErrors((prev) => ({ ...prev, [name]: undefined }));
    }
  };

  const handleToggleChange = (name: keyof FormState, value: boolean) => {
    setFormState((prev) => ({ ...prev, [name]: value }));
  };

  const validateForm = (): boolean => {
    const newErrors: Partial<Record<keyof FormState, string>> = {};

    if (!formState.size) newErrors.size = t.required;
    if (!formState.rooms) newErrors.rooms = t.required;
    if (!formState.floor) newErrors.floor = t.required;
    if (!formState.totalFloors) newErrors.totalFloors = t.required;
    if (!formState.neighbourhood) newErrors.neighbourhood = t.required;

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();

    if (!validateForm()) return;

    setIsLoading(true);
    setShowResult(false);

    try {
      const floor = parseInt(formState.floor);
      const totalFloors = parseInt(formState.totalFloors);

      const data = await callInferenceModel({
        size_m2: parseInt(formState.size),
        nr_of_rooms: parseInt(formState.rooms),
        floor: floor,
        building_total_floors: totalFloors,
        neighbourhood: formState.neighbourhood,
        is_first_floor: floor === 1 ? 1 : 0,
        is_last_floor: floor === totalFloors ? 1 : 0,
        near_public_transport: formState.nearTransport ? 1 : 0,
        is_furnished: formState.furnished ? 1 : 0,
      });

      setResult({ price_low: data.lower_bound, price_high: data.upper_bound });
      setTimeout(() => setShowResult(true), 50);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="min-h-screen flex items-center justify-center px-4 py-12 bg-[#FAF8F5]">
      <div className="w-full max-w-[680px] bg-white p-8 md:p-12 rounded-lg border border-[#E0DCD7]">
        {/* Header with Language Switcher */}
        <header className="mb-8">
          <div className="flex justify-end mb-4">
            <LanguageSwitcher language={language} onChange={setLanguage} />
          </div>
          <div className="text-center">
            <p className="text-xs uppercase tracking-[0.2em] text-[#7D8B7A] font-medium mb-3">
              {t.header}
            </p>
            <h1 className="font-serif text-3xl md:text-4xl text-[#1C1C1E] leading-tight text-balance">
              {t.subtitle}
            </h1>
          </div>
        </header>

        {/* Divider */}
        <hr className="border-t border-[#E0DCD7] mb-8" />

        {/* Form */}
        <form onSubmit={handleSubmit}>
          {/* Input Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-5">
            <Field
              label={t.size}
              name="size"
              type="number"
              value={formState.size}
              onChange={handleFieldChange}
              placeholder="e.g. 100"
              error={errors.size}
            />
            <Field
              label={t.rooms}
              name="rooms"
              type="number"
              value={formState.rooms}
              onChange={handleFieldChange}
              placeholder="e.g. 4"
              error={errors.rooms}
            />
            <Field
              label={t.floor}
              name="floor"
              type="number"
              value={formState.floor}
              onChange={handleFieldChange}
              placeholder="e.g. 2"
              error={errors.floor}
            />
            <Field
              label={t.totalFloors}
              name="totalFloors"
              type="number"
              value={formState.totalFloors}
              onChange={handleFieldChange}
              placeholder="e.g. 5"
              error={errors.totalFloors}
            />
            <div className="md:col-span-2">
              <Field
                label={t.neighbourhood}
                name="neighbourhood"
                type="select"
                value={formState.neighbourhood}
                onChange={handleFieldChange}
                options={t.neighbourhoods}
                error={errors.neighbourhood}
                selectPlaceholder={t.select}
              />
            </div>
          </div>

          {/* Toggle Switches */}
          <div className="mt-6 border-t border-[#E0DCD7] pt-6">
            <Toggle
              label={t.furnished}
              name="furnished"
              checked={formState.furnished}
              onChange={handleToggleChange}
            />
            <Toggle
              label={t.nearTransport}
              name="nearTransport"
              checked={formState.nearTransport}
              onChange={handleToggleChange}
            />
          </div>

          {/* Submit Button */}
          <button
            type="submit"
            disabled={isLoading}
            className={`
              w-full mt-8 py-4 
              bg-[#1C1C1E] text-white 
              text-sm font-medium uppercase tracking-[0.1em]
              rounded-md
              transition-colors duration-200
              hover:bg-[#7D8B7A]
              disabled:opacity-70 disabled:cursor-not-allowed
            `}
          >
            {isLoading ? t.calculating : t.submit}
          </button>
        </form>

        {/* Result Section */}
        {result && <ResultCard result={result} visible={showResult} t={t} />}
      </div>
    </main>
  );
}
