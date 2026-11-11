import { CatLogo } from "@/components/cat-logo"
import { Composer } from "@/components/composer"

/** New-chat screen: a greeting and a centered composer. */
export function Welcome() {
  return (
    <div className="flex min-h-0 flex-1 flex-col items-center justify-center overflow-y-auto px-4 pb-[12vh]">
      <div className="flex w-full max-w-[44rem] flex-col items-center">
        <div className="flex flex-col items-center animate-in fade-in-0 slide-in-from-bottom-2 duration-500 ease-snappy">
          <CatLogo className="mb-5 size-11 rounded-[14px]" />
          <h1 className="mb-6 text-center text-[1.75rem] leading-tight font-semibold tracking-[-0.035em] text-balance sm:text-[2.1rem]">
            What's on your mind?
          </h1>
        </div>
        <Composer autoFocus centered />
      </div>
    </div>
  )
}
